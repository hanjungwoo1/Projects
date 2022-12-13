import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def smtp_setting(email: str, password: str, type: str):
    mail_type = None
    port = 587

    if type == 'naver':
        mail_type = 'smtp.naver.com'
    elif type == 'gmail':
        mail_type = 'smtp.gmail.com'
    else:
        mail_type = 'smtp.gmail.com'

    # SMTP 세션 생성
    smtp = smtplib.SMTP(mail_type, port)
    smtp.set_debuglevel(True)

    # SMTP 계정 인증 설정
    smtp.ehlo()
    smtp.starttls()  # TLS 사용시 호출
    smtp.login(email, password)

    return smtp


def send_plain_mail(sender, receiver, email, password, subject, content):
    try:
        # SMTP 세션 생성
        smtp = smtp_setting(email, password, "gmail")

        # 이메일 데이터 설정
        msg = MIMEText(content)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["receiver"] = receiver

        # 메일 전송
        smtp.sendmail(sender, receiver, msg.as_string())

    except Exception as e:
        print("error", e)

    finally:
        if smtp is not None:
            smtp.quit()

target = input("input Target : ")
sender = input("input Sender's ID : ")
password = input("input Sender's password : ")

send_plain_mail(sender,
                target,
                sender,
                password,
                "얌망",
                "얌망"
                )