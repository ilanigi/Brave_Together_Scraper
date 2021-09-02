import re

def stripToTextVector(text):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove Emails
    text = re.sub('\S*@\S*\s?', '', text)

    # Remove new line characters
    text = re.sub('\s+', ' ', text)

    # Remove distracting single quotes
    text = re.sub("\'", "", text)


    #Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    #Remove punctuations
    text = ''.join(u for u in text if u not in ("?", ".", ";", ":", "!",'”','"',"[","]","{","}","(",")"))

    #Replace slashes with spaces
    text = text.replace("/"," ")

    #Change string to lower letters
    text = text.lower()

    return text.split()
