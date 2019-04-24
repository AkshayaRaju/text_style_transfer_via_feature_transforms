# which helps format the raw dialog data
import unicodedata
import codecs
import re

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# lower case and 
def normalize_str(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"(\.\.\.)", r"\1 ", s)
    # s = re.sub(r"([.?!])", r" \1", s) # replace "." with " ." and so on
    s = re.sub(r"[^a-zA-Z.?!\"_<>\']+", r" ", s)
    s = re.sub(r"<unk>", "<UNK>", s) # make the unk upper case again
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class DataLoader(object):
    # Splits each line of the file into a dictionary of fields
    @staticmethod
    def loadLines(fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines


    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    @staticmethod
    def loadConversations(fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = eval(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations


    # Extracts pairs of sentences from conversations
    @staticmethod
    def extractSentencePairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs
