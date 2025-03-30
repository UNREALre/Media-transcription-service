import logging
from typing import List, Optional

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


async def send_email(
        email_to: str,
        subject: str,
        html_content: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None
) -> bool:
    """Send email using configured SMTP server"""
    message = MIMEMultipart()
    message["From"] = settings.EMAIL_FROM
    message["To"] = email_to
    message["Subject"] = subject

    if cc:
        message["Cc"] = ", ".join(cc)
    if bcc:
        message["Bcc"] = ", ".join(bcc)

    message.attach(MIMEText(html_content, "html"))

    try:
        smtp = aiosmtplib.SMTP(
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            use_tls=settings.SMTP_TLS
        )

        await smtp.connect()

        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            await smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)

        await smtp.send_message(message)
        await smtp.quit()

        logger.info(f"Email sent to {email_to}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False


async def send_notification_email(
        email_to: str,
        subject: str,
        message: str
) -> bool:
    """Send notification email with standardized format"""
    html_content = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ padding: 20px; max-width: 600px; margin: 0 auto; }}
                .header {{ background-color: #4a76a8; color: white; padding: 10px 20px; }}
                .content {{ padding: 20px; border: 1px solid #ddd; }}
                .footer {{ color: #777; font-size: 12px; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{settings.APP_NAME}</h2>
                </div>
                <div class="content">
                    <p>{message}</p>
                </div>
                <div class="footer">
                    <p>© {settings.APP_NAME}</p>
                </div>
            </div>
        </body>
    </html>
    """

    return await send_email(email_to, subject, html_content)
