'''
Evaluate the type of a string
'''

def evaluate(string, string_type):

    if string_type=='FLOAT':

        return float(string)

    if string_type=='INT':

        return int(string)

    if string_type=='STRING':

        return string

    if string_type=='BOOL':

        if string=='True':

            return True

        else:

            return False
