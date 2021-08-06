import telebot,subprocess
from datetime import datetime

# Ingresar el token de su bot en la linea #5 reemplazar la palabra 'TOKEN'
bot = telebot.TeleBot('TOKEN')
@bot.message_handler(content_types=['voice', 'audio'])

def get_audio_messages(message):
    audioentradas='audioentradas\\'
    audioconvertidos='audioconvertidos\\'

    file_info = bot.get_file(message.voice.file_id)
    #descarga el archivo audio
    downloaded_file = bot.download_file(file_info.file_path)
    #crea e archivo en la ruta base con un nombre especifico
    nombrearchivosinextension=datetime.today().strftime('%Y-%m-%d %H%M%S')
    nombrearchivo =nombrearchivosinextension +'.ogg'
    with open(audioentradas+nombrearchivo, 'wb') as new_file:
        new_file.write(downloaded_file)

    src_filename = audioentradas+nombrearchivo
    dest_filename =audioconvertidos+nombrearchivosinextension+'.wav'

    process = subprocess.run(['ffmpeg\\bin\\ffmpeg.exe', '-i', src_filename, dest_filename])

    msgsentimiento=ProcesarIAReconocimientoSentimientos(dest_filename)

    text=msgsentimiento
    bot.send_message(message.chat.id, text)   

def ProcesarIAReconocimientoSentimientos(urlWAV):
    return 'se guardo el archivo como '+urlWAV

bot.polling(none_stop = True)
