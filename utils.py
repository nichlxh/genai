import time
import re
def cleanOutput (message):
    '''
    Function to prevent latex formatting during input and output on Streamlit
    :param message: input message to clean
    :return: message with fixed formatting
    '''
    # ignore the warnings.
    # replacements to prevent Latex formatting from both system and user side.
    if '\$' in message: #system
        message = message.replace('\$', '\\$')
    elif '$' in message: # user
        message = message.replace('$', '\\$')

    return message

def stream_data(msg):
    '''
    Function to print output with streaming typewriter effect.
    :param msg: input message to stream to output
    :yield:  each character for the effect.
    '''
    msg = cleanOutput(msg)
    for char in msg:
        yield char
        time.sleep(0.01)

def extractCitations(message):
    '''
    Function to extract citation numbers from e.g. [1, 2, 3]
    :param message: input message to extract citation numbers
    :return: matches - citation number strings found in message
             numbers - extracted unique citation numbers
    '''
    pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    # Find all matches
    matches = re.findall(pattern, message)

    # Process matches to extract individual numbers
    numbers = [int(num) for group in matches for num in re.split(r',\s*', group)]
    numbers = sorted(list(set(numbers)))

    return matches, numbers