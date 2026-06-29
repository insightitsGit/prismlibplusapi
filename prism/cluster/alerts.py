"""
prism.cluster.alerts — AlertManager: health alerts + admin email notifications.

Every node runs an AlertManager. When a health check fails, a token budget
threshold is crossed, a security signal fires, or an app emits a critical
event — the AlertManager:

  1. Evaluates the alert against configured rules
  2. Broadcasts a SIGNAL frame via the CHORUS tunnel (all nodes aware)
  3. Sends an email to the configured admin address(es)

Mail service options (configure one — no external agent needed):
  - SMTP       : any SMTP server (Gmail, Outlook, corporate)
  - SendGrid   : pip install sendgrid
  - Mailgun    : HTTP API, no extra package
  - AWS SES    : pip install boto3
  - Resend     : pip install resend (modern, simple)

Zero dependencies by default — SMTP works with Python stdlib smtplib.
Install extras only if you want a managed provider:
  pip install "prismlib[alerts-sendgrid]"
  pip install "prismlib[alerts-ses]"
  pip install "prismlib[alerts-resend]"
"""

from __future__ import annotations

import asyncio
import logging
import operator
import re
import smtplib
import ssl
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert levels and rules
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """
    A rule that fires when a condition is met.

    Examples:
        AlertRule("cpu_high",     "cpu_pct > 90",       AlertLevel.WARNING)
        AlertRule("ram_critical", "ram_used_pct > 95",  AlertLevel.CRITICAL)
        AlertRule("index_empty",  "index_size == 0",    AlertLevel.WARNING)
        AlertRule("budget_80pct", "budget_used_pct > 80", AlertLevel.WARNING)
        AlertRule("sub_errors",   "sub_errors > 10",    AlertLevel.CRITICAL)
    """
    name:       str
    condition:  str          # Comparison expression, e.g. "cpu_pct > 90"
    level:      AlertLevel
    cooldown_s: float = 300  # don't re-fire same alert within this many seconds
    _last_fired: float = field(default=0.0, init=False, repr=False)

    def should_fire(self, context: dict) -> bool:
        try:
            result = _evaluate_condition(self.condition, context)
        except Exception:
            return False
        if result and (time.time() - self._last_fired) > self.cooldown_s:
            self._last_fired = time.time()
            return True
        return False


_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_COND_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$"
)


def _parse_literal(raw: str) -> bool | int | float:
    s = raw.strip()
    if s == "True":
        return True
    if s == "False":
        return False
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if "." in s:
        return float(s)
    return int(s)


def _evaluate_condition(condition: str, context: dict) -> bool:
    """Safe comparison evaluator — no arbitrary code execution."""
    m = _COND_RE.match(condition)
    if not m:
        return False
    key, op_sym, rhs_raw = m.groups()
    if key not in context:
        return False
    left = context[key]
    right = _parse_literal(rhs_raw)
    op_fn = _OPS.get(op_sym)
    if op_fn is None:
        return False
    return bool(op_fn(left, right))


# Default rules applied to every node
DEFAULT_RULES: list[AlertRule] = [
    AlertRule("cpu_critical",     "cpu_pct > 95",           AlertLevel.CRITICAL, cooldown_s=120),
    AlertRule("cpu_high",         "cpu_pct > 85",           AlertLevel.WARNING,  cooldown_s=300),
    AlertRule("ram_critical",     "ram_used_pct > 95",      AlertLevel.CRITICAL, cooldown_s=120),
    AlertRule("ram_high",         "ram_used_pct > 85",      AlertLevel.WARNING,  cooldown_s=300),
    AlertRule("disk_critical",    "disk_used_pct > 95",     AlertLevel.CRITICAL, cooldown_s=600),
    AlertRule("disk_high",        "disk_used_pct > 80",     AlertLevel.WARNING,  cooldown_s=600),
    AlertRule("index_empty",      "index_size == 0",        AlertLevel.WARNING,  cooldown_s=300),
    AlertRule("sub_errors_high",  "sub_errors > 5",         AlertLevel.WARNING,  cooldown_s=300),
    AlertRule("sub_loop_dead",    "sub_task_running == False", AlertLevel.CRITICAL, cooldown_s=60),
    AlertRule("latency_high",     "avg_latency_ms > 500",   AlertLevel.WARNING,  cooldown_s=300),
    AlertRule("budget_80pct",     "budget_used_pct > 80",   AlertLevel.WARNING,  cooldown_s=3600),
    AlertRule("budget_95pct",     "budget_used_pct > 95",   AlertLevel.CRITICAL, cooldown_s=1800),
]


# ---------------------------------------------------------------------------
# Mail backends
# ---------------------------------------------------------------------------

@dataclass
class SMTPConfig:
    """
    Standard SMTP — works with Gmail, Outlook, corporate mail, or self-hosted.

    Gmail quick-start:
        host     = "smtp.gmail.com"
        port     = 587
        username = "you@gmail.com"
        password = "your-app-password"   # NOT your login password
        # Generate at: myaccount.google.com → Security → App passwords
    """
    host:       str
    port:       int   = 587
    username:   str   = ""
    password:   str   = ""
    use_tls:    bool  = True
    from_addr:  str   = ""

    @property
    def sender(self) -> str:
        return self.from_addr or self.username


@dataclass
class SendGridConfig:
    api_key:    str
    from_addr:  str   = "alerts@prismlib.io"


@dataclass
class MailgunConfig:
    api_key:    str
    domain:     str
    from_addr:  str   = ""

    @property
    def sender(self) -> str:
        return self.from_addr or f"alerts@{self.domain}"


@dataclass
class SESConfig:
    region:     str   = "us-east-1"
    from_addr:  str   = ""


@dataclass
class ResendConfig:
    api_key:    str
    from_addr:  str   = "alerts@prismlib.io"


MailConfig = SMTPConfig | SendGridConfig | MailgunConfig | SESConfig | ResendConfig


# ---------------------------------------------------------------------------
# Email renderer
# ---------------------------------------------------------------------------

def _render_email(
    node_id:    str,
    level:      str,
    event_type: str,
    title:      str,
    message:    str,
    data:       dict,
) -> tuple[str, str, str]:
    """Returns (subject, plain_text, html)."""

    level_upper = level.upper()
    emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level_upper, "🔔")
    subject = f"{emoji} [{level_upper}] PrismLib — {title} (node: {node_id})"

    data_rows = "\n".join(f"  {k}: {v}" for k, v in data.items())
    plain = f"""
PrismLib Cluster Alert
======================
Level:      {level_upper}
Event:      {event_type}
Node:       {node_id}
Time:       {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

{message}

Details:
{data_rows}

---
Sent by PrismLib AlertManager.
To configure alerts: https://github.com/insightitsGit/prismlib
""".strip()

    data_html = "".join(
        f"<tr><td style='padding:4px 12px;color:#888'>{k}</td>"
        f"<td style='padding:4px 12px'>{v}</td></tr>"
        for k, v in data.items()
    )
    color = {"INFO": "#3b82f6", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}.get(level_upper, "#6366f1")

    html = f"""
<!DOCTYPE html>
<html>
<body style="font-family:system-ui,sans-serif;background:#0f0f0f;color:#e5e7eb;padding:32px">
  <div style="max-width:600px;margin:0 auto;background:#1a1a1a;border-radius:8px;overflow:hidden">
    <div style="background:{color};padding:20px 24px">
      <h1 style="margin:0;font-size:18px;color:#fff">{emoji} {title}</h1>
      <p style="margin:4px 0 0;opacity:0.85;font-size:13px;color:#fff">
        {level_upper} · {event_type} · node: {node_id}
      </p>
    </div>
    <div style="padding:24px">
      <p style="margin:0 0 20px;line-height:1.6">{message}</p>
      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <tr style="background:#252525">
          <th style="padding:6px 12px;text-align:left;color:#9ca3af">Key</th>
          <th style="padding:6px 12px;text-align:left;color:#9ca3af">Value</th>
        </tr>
        {data_html}
      </table>
      <p style="margin:20px 0 0;font-size:12px;color:#6b7280">
        Sent {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by
        <a href="https://github.com/insightitsGit/prismlib" style="color:#6366f1">PrismLib AlertManager</a>
      </p>
    </div>
  </div>
</body>
</html>
""".strip()

    return subject, plain, html


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Evaluates health/status dicts against rules, fires alerts via email
    and CHORUS SIGNAL frames.

    Quick-start (Gmail SMTP):
        alerter = AlertManager(
            node_id     = "node-a",
            admin_email = ["ops@yourcompany.com"],
            mail_config = SMTPConfig(
                host     = "smtp.gmail.com",
                username = "you@gmail.com",
                password = "your-app-password",
            ),
        )

    Quick-start (Resend — modern API, free tier 3k emails/mo):
        alerter = AlertManager(
            node_id     = "node-a",
            admin_email = ["ops@yourcompany.com"],
            mail_config = ResendConfig(api_key="re_xxx", from_addr="alerts@yourdomain.com"),
        )
    """

    def __init__(
        self,
        node_id:     str,
        admin_email: list[str],
        mail_config: Optional[MailConfig] = None,
        rules:       Optional[list[AlertRule]] = None,
        fabric:      Optional[Any] = None,   # CHORUSFabric — broadcast signals
        min_level:   AlertLevel = AlertLevel.WARNING,
    ) -> None:
        self.node_id     = node_id
        self.admin_email = admin_email
        self._mail       = mail_config
        self._rules      = rules if rules is not None else DEFAULT_RULES
        self._fabric     = fabric
        self._min_level  = min_level
        self._fired:     list[dict] = []   # alert history

    # ------------------------------------------------------------------
    # Health evaluation — called by HealthMonitor every heartbeat
    # ------------------------------------------------------------------

    async def evaluate_health(self, health_dict: dict) -> None:
        """
        Evaluate a health snapshot against all rules.
        Fires alerts for any rule whose condition is met.

        health_dict keys: cpu_pct, ram_used_pct, disk_used_pct,
                          index_size, sub_errors, sub_task_running,
                          avg_latency_ms, budget_used_pct, ...
        """
        # Compute derived fields
        ctx = {**health_dict}
        ctx.setdefault("ram_used_pct",
            health_dict.get("ram_used_mb", 0) /
            max(health_dict.get("ram_total_mb", 1), 1) * 100
        )
        ctx.setdefault("disk_used_pct",
            health_dict.get("disk_used_gb", 0) /
            max(health_dict.get("disk_total_gb", 1), 1) * 100
        )

        for rule in self._rules:
            if rule.should_fire(ctx):
                await self.send_alert(
                    level      = rule.level.value,
                    event_type = rule.name,
                    title      = f"Alert: {rule.name.replace('_', ' ').title()}",
                    message    = (
                        f"Rule '{rule.name}' fired on node {self.node_id}. "
                        f"Condition: {rule.condition}."
                    ),
                    data       = {k: v for k, v in ctx.items()
                                  if isinstance(v, (int, float, bool, str))},
                )

    # ------------------------------------------------------------------
    # Main send_alert — email + SIGNAL frame
    # ------------------------------------------------------------------

    async def send_alert(
        self,
        level:      str,
        event_type: str,
        title:      str,
        message:    str,
        data:       Optional[dict] = None,
    ) -> None:
        """
        Send an alert:
          1. Broadcast SIGNAL frame via CHORUS tunnel (instant, all nodes)
          2. Send email to all admin_email addresses
        """
        data = data or {}

        # Skip below minimum level
        level_order = {AlertLevel.INFO: 0, AlertLevel.WARNING: 1, AlertLevel.CRITICAL: 2}
        if level_order.get(AlertLevel(level), 0) < level_order.get(self._min_level, 0):
            return

        record = {
            "node_id":    self.node_id,
            "level":      level,
            "event_type": event_type,
            "title":      title,
            "message":    message,
            "data":       data,
            "ts":         time.time(),
        }
        self._fired.append(record)
        logger.warning("[%s] ALERT [%s] %s: %s", self.node_id, level.upper(), event_type, title)

        # 1. Broadcast via CHORUS tunnel so all nodes are aware
        if self._fabric is not None:
            try:
                from prism.lib.fabric import SignalPayload
                signal = SignalPayload(
                    node_id     = self.node_id,
                    signal_type = event_type,
                    severity    = level,
                    description = message,
                    data        = {k: str(v) for k, v in data.items()},
                )
                await self._fabric.emit_signal(signal)
            except Exception as exc:
                logger.debug("Alert SIGNAL broadcast failed: %s", exc)

        # 2. Send email
        if self._mail and self.admin_email:
            subject, plain, html = _render_email(
                self.node_id, level, event_type, title, message, data
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._send_email_sync,
                subject, plain, html,
            )

    # ------------------------------------------------------------------
    # Mail backends (sync — run in executor to not block async loop)
    # ------------------------------------------------------------------

    def _send_email_sync(self, subject: str, plain: str, html: str) -> None:
        try:
            if isinstance(self._mail, SMTPConfig):
                self._send_smtp(subject, plain, html)
            elif isinstance(self._mail, SendGridConfig):
                self._send_sendgrid(subject, plain, html)
            elif isinstance(self._mail, MailgunConfig):
                self._send_mailgun(subject, plain, html)
            elif isinstance(self._mail, SESConfig):
                self._send_ses(subject, plain, html)
            elif isinstance(self._mail, ResendConfig):
                self._send_resend(subject, plain, html)
            logger.info("[%s] Alert email sent to %s", self.node_id, self.admin_email)
        except Exception as exc:
            logger.error("[%s] Alert email FAILED: %s", self.node_id, exc)

    def _send_smtp(self, subject: str, plain: str, html: str) -> None:
        cfg = self._mail
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg.sender
        msg["To"]      = ", ".join(self.admin_email)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(cfg.host, cfg.port) as server:
            if cfg.use_tls:
                server.starttls(context=context)
            if cfg.username and cfg.password:
                server.login(cfg.username, cfg.password)
            server.sendmail(cfg.sender, self.admin_email, msg.as_string())

    def _send_sendgrid(self, subject: str, plain: str, html: str) -> None:
        try:
            import sendgrid
            from sendgrid.helpers.mail import Mail
        except ImportError:
            raise ImportError("pip install sendgrid")
        cfg = self._mail
        sg  = sendgrid.SendGridAPIClient(api_key=cfg.api_key)
        for to in self.admin_email:
            mail = Mail(
                from_email    = cfg.from_addr,
                to_emails     = to,
                subject       = subject,
                html_content  = html,
            )
            sg.send(mail)

    def _send_mailgun(self, subject: str, plain: str, html: str) -> None:
        import urllib.request
        import urllib.parse
        cfg = self._mail
        url = f"https://api.mailgun.net/v3/{cfg.domain}/messages"
        data = urllib.parse.urlencode({
            "from":    cfg.sender,
            "to":      ", ".join(self.admin_email),
            "subject": subject,
            "text":    plain,
            "html":    html,
        }).encode()
        import base64
        creds = base64.b64encode(f"api:{cfg.api_key}".encode()).decode()
        req   = urllib.request.Request(url, data=data,
                    headers={"Authorization": f"Basic {creds}"})
        urllib.request.urlopen(req)

    def _send_ses(self, subject: str, plain: str, html: str) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError("pip install boto3")
        cfg = self._mail
        ses = boto3.client("ses", region_name=cfg.region)
        ses.send_email(
            Source      = cfg.from_addr,
            Destination = {"ToAddresses": self.admin_email},
            Message     = {
                "Subject": {"Data": subject},
                "Body":    {
                    "Text": {"Data": plain},
                    "Html": {"Data": html},
                },
            },
        )

    def _send_resend(self, subject: str, plain: str, html: str) -> None:
        try:
            import resend
        except ImportError:
            raise ImportError("pip install resend")
        cfg = self._mail
        resend.api_key = cfg.api_key
        resend.Emails.send({
            "from":    cfg.from_addr,
            "to":      self.admin_email,
            "subject": subject,
            "html":    html,
            "text":    plain,
        })

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def alert_history(self) -> list[dict]:
        return list(self._fired[-50:])   # last 50 alerts

    @property
    def status(self) -> dict:
        return {
            "node_id":      self.node_id,
            "admin_email":  self.admin_email,
            "mail_backend": type(self._mail).__name__ if self._mail else "none",
            "rules_count":  len(self._rules),
            "alerts_fired": len(self._fired),
            "last_alert":   self._fired[-1] if self._fired else None,
        }
