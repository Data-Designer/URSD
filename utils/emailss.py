import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_email(epoch,metric1,metric2,metric3,metric4):
    """[],[],[],[]->string"""
    content = list(zip(epoch,metric1,metric2,metric3,metric4)) #list tuple
    message = '\n'.join([str(i) for i in content])
    message = MIMEText(message)   # email content


    message['From'] = Header('Most Lover Chuang')   # email sender
    message['To'] = Header('Most Lover Hui')   # email receiver
    message['Subject'] = Header('Model Performance--weight decay0.01--lr5e-5--batch64')   # theme

    mail = smtplib.SMTP()
    mail.connect("smtp.qq.com")
    mail.login("******", "******")
    mail.sendmail("******", ["******"], message.as_string())


if __name__ == '__main__':
    epoch, metric1, metric2, metric3, metric4 = [1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]
    send_email(epoch, metric1, metric2, metric3, metric4)
    print('Email Done!')
