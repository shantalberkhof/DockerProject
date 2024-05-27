import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile

#import yolo5.app
#from polybot.img_proc import Img
import requests
import boto3
from loguru import logger
import json


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            path = self.download_user_photo(msg)
            file_name = os.path.basename(path)
            #my_img = Img(path)

            # TODO upload the photo to S3

            # Initialize the S3 client
            s3 = boto3.client('s3')

            # Specify the S3 bucket name and the key (path) under which the object will be stored
            bucket_name = 'shantal-dockerproject'
            object_key = f'data/{file_name}'

            # Specify the local file path to be uploaded to S3
            local_file_path = f'{path}'

            # Upload the object to S3
            s3.upload_file(local_file_path, bucket_name, object_key)
            logger.info(f'prediction: {file_name}. was upload to s3 successfully')

            # TODO send an HTTP request to the `yolo5` service for prediction
            """ THIS IS THE OLD VERSION
            # Send an HTTP request to the `yolo5` service for prediction
            json_data = yolo5.app.predict()
            # Send the returned results to the Telegram end-user
            self.send_text(msg['chat']['id'], json_data)

            yolo5_url = 'http://yolo5-service-endpoint/predict'
            path = '/home/shantalberkhof/PycharmProjects/DockerProject/local_file.jpg'
            files = {'photo': open(path, 'rb')}
            response = requests.post(yolo5_url, files=files)
            response.raise_for_status()
            detection_results = response.json()
            """

            yolo5_base_url = f"http://yolo5:8081/predict"
            yolo5_url = f"{yolo5_base_url}?imgName={object_key}"
            logger.info(f'Calling the yolo service')
            response = requests.post(yolo5_url)

            #json_data = response.json()
            # TODO send the returned results to the Telegram end-user
            # turn response.text into something normal - Not required.
            # A change
            #json_data = response.json()
            #data = json_data
            #object_counts = {}

            # for label in data['labels']:
            #     object_class = label['class']
            #     if object_class in object_counts:
            #         object_counts[object_class] += 1
            #     else:
            #         object_counts[object_class] = 1
            #
            # summary_string = "Detected objects:\n"
            # for object_class, count in object_counts.items():
            #     summary_string += f"{object_class}: {count}\n"


            # Send the returned results to the Telegram end-user (was summary_string.text)
            self.send_text(msg['chat']['id'], response.text)

            #self.send_text(msg['chat']['id'], response.text)

            #self.send_message(msg['chat']['id'], f'Detection results: {response.text}')
