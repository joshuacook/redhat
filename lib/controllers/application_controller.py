def help_me(commands):
    print("Available commands: ")
    for command in commands:
        print(command)

def quit(verification):
    if verification == "Y" or verification == "Yes":
        print("Thank you.\n")
        exit(0)

def reload_all(verification):
    if verification == "Y" or verification == "Yes":
        print("Just reloaded everything.\n")
