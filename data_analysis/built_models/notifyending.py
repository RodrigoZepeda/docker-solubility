import json
import telegram
def notify_ending(message):
    bot = telegram.Bot(token='884880200:AAFXIlerG9cVfGXGER_FaEPdKDW92o1W_b8')
    bot.sendMessage(chat_id='358528401', text=message)
