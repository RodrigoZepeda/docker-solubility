import json
import telegram
def notify_ending(message):
    bot = telegram.Bot(token='884880200:AAFrIUC-f4Z_0J_koQGX8dQl5qjmuTG7XS8')
    bot.sendMessage(chat_id='358528401', text=message)
