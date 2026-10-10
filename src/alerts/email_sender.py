import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT


def send_email(subject, html_body):
    """Envoie un email HTML."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("[email] Identifiants manquants (EMAIL_SENDER, EMAIL_PASSWORD)")
        return False

    recipient = EMAIL_RECIPIENT or EMAIL_SENDER

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, recipient, msg.as_string())
        print(f"[email] Envoye a {recipient}")
        return True
    except Exception as ex:
        print(f"[email] Erreur : {ex}")
        return False


def format_news_email(news_item):
    """Formate un email HTML pour une news."""
    sent = news_item["sentiment"]
    emoji = "🟢" if sent > 0.2 else ("🔴" if sent < -0.2 else "⚪")
    color = "#27AE60" if sent > 0.2 else ("#E74C3C" if sent < -0.2 else "#95A5A6")

    subject = f"{emoji} [{news_item['tickers']}] {news_item['title'][:80]}"

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 700px;">
        <div style="background: {color}; padding: 15px; color: white; border-radius: 8px;">
            <h2 style="margin: 0;">{emoji} {news_item['tickers']}</h2>
            <p style="margin: 5px 0 0 0; font-size: 14px;">Alerte news importante</p>
        </div>

        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px; margin-top: 15px;">
            <h3 style="margin-top: 0;">📰 {news_item['title']}</h3>
            <p><b>Source :</b> {news_item.get('source', 'N/A')}</p>
            <p><b>Date :</b> {news_item['date']}</p>
        </div>

        <div style="padding: 15px; background: white; border-left: 4px solid {color}; margin-top: 15px;">
            <h3>📊 Analyse</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px;"><b>Sentiment</b></td>
                    <td style="padding: 8px;"><span style="color: {color}; font-size: 20px; font-weight: bold;">{sent:+.3f}</span></td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>Type d'evenement</b></td>
                    <td style="padding: 8px;">{news_item.get('event_type', '?')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>Impact estime</b></td>
                    <td style="padding: 8px;">{news_item.get('impact', '?')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><b>Raison</b></td>
                    <td style="padding: 8px;">{news_item.get('raison', 'N/A')}</td>
                </tr>
            </table>
        </div>

        <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; font-size: 12px;">
            ⚠️ <b>Outil pedagogique.</b> Ne constitue pas un conseil en investissement.
        </div>
    </body>
    </html>
    """
    return subject, html


def send_news_alert(news_item):
    """Envoie une alerte email pour une news."""
    subject, html = format_news_email(news_item)
    return send_email(subject, html)


if __name__ == "__main__":
    # Test
    test_news = {
        "tickers": "BIAT",
        "title": "Test - BIAT annonce des resultats en hausse",
        "sentiment": 0.75,
        "event_type": "resultats",
        "impact": "haussier",
        "raison": "Resultats meilleurs que prevu",
        "source": "Test",
        "date": "2026-10-09",
    }
    print("Envoi email test...")
    send_news_alert(test_news)