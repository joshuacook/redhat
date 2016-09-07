def test_function():
    pass

def validated_input(choices):
    input_value = None
    while input_value == None:
        try:
            input_value = input("$ ")
            if input_value not in choices:
                print("Invalid command: {}".format(input_value))
                input_value = None
        except SyntaxError:
            print("Invalid command:")
            input_value = None
    return input_value
