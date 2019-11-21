
import sys
import os
import data
import recognition
import GAN
import tensorflow as tf
import  smtplib
from email.message import EmailMessage

def send_email():
    with open('email') as fp:
        # Create a text/plain message
        msg = EmailMessage()
        msg.set_content(fp.read())

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = 'GAN File finished running'
    msg['From'] = 'mcclain.thiel@gmail.com'
    msg['To'] = 'mcclain.thiel@berkeley.edu'

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


#main function
if __name__ =='__main__':
    """
    TODO:
    Add email alert 
    """
    print('Usage: python3 main.py homogeneous_data_to_be_replicated baseline_data')
    print('Tensorflow Version: ', tf.__version__)
    imgs_to_generated = sys.argv[1:][0]
    base_line_imgs = sys.argv[1:][1]
    gen_data = data.Data(imgs_to_generated, base_line_imgs).data_array
    gan_instance = GAN.GAN()
    gan_instance.train(gen_data, 3)

    send_email()



