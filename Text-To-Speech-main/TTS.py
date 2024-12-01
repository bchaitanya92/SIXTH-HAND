# import library
import PyPDF2
import pyttsx3

# if any error when installation install the dependency library, If you receive errors such as No module named win32com.client, No module named win32, or No module named win32api, you will need to additionally install pypiwin32.
engine = pyttsx3.init()  # declare

def myfun():
    # path of the PDF file
    path = open('C:\\Selecao\\Text-To-Speech-main\\file.pdf', 'r')

    # creating a PdfFileReader object
    pdfReader = PyPDF2.PdfFileReader(path)

    # the page with which you want to start
    # this will read the page of 25th page.
    from_page = pdfReader.getPage(0)

    # extracting the text from the PDF
    text = from_page.extractText()

    # reading the text
    speak = pyttsx3.init()
    speak.say(text)
    speak.runAndWait()


def text():
    engine = pyttsx3.init()  # object creation
    read = input("Write The Text :\n")

    engine.say(read)  # pass any string
    engine.runAndWait()  # output in voice format
    """ RATE"""
    # Changing Voice , Rate and Volume
    rate = engine.getProperty('rate')  # getting details of current speaking rate
    print(rate)  # printing current voice rate
    engine.setProperty('rate', 125)  # setting up new voice rate

    #####VOLUME#####
    volume = engine.getProperty('volume')  # getting to know current volume level (min=0 and max=1)
    print(volume)  # printing current volume level
    engine.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1

    #####VOICE#####
    voices = engine.getProperty('voices')  # getting details of current voice
    # engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female

    # engine.say('My current speaking rate is ' + str(rate))
    engine.runAndWait()
    engine.stop()

    #####Saving Voice to a file#####
    # On linux make sure that 'espeak' and 'ffmpeg' are installed
    engine.save_to_file('Hello World', 'test.mp3')
    engine.runAndWait()


print("Enter Your Choice")
choice = int(input("1: Read From Text\n"))

if choice == 1:

    text()

else:
    myfun()
