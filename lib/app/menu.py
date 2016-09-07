from builtins import input
from IPython import embed
from os import system
from redis import StrictRedis
from rq import Queue

from lib.helpers.application_helper import validated_input
class Menu:
    '''Display a menu and respond to choices when run.'''
    def __init__(self):
        system('clear')

        self.queued_actions = ['heartbeat']

        self.choices = {
            'embed' : embed,
            'help' : help_me,
            'quit' : quit,
        }

        self.prompts = {
            'embed' : None,
            'heartbeat' : None,
            'help' : {
                'commands' : ['continue', self.choices.keys()]
            },
            'quit' : {
                'verification' : ['Are you sure you want to quit? [Y/n]', 'Y']
            },
            'reload' : {
                'verification' : ['Are you sure you want to reload all modules? [Y/n]', 'Y']
            }
        }

    def __repr__(self):
        menu  = '               Backend Interface\n'
        menu += '          ============================\n'
        menu += '                                      \n'
        return menu

    def run(self):
        action = None # Default initial action and args
        args = None
        while True:
            self.action_runner(action, args)
            command = validated_input(self.choices)
            action = self.choices.get(command)
            args = self.arg_parser(command)

    def action_runner(self, action, args):
        system('clear')
        print(self)
        if action:
            if action == embed:
                action()
            elif action in self.queued_actions:
                if args:
                    Q.enqueue(action, **args)
                else:
                    Q.enqueue(action)
            else:
                if args:
                    action(**args)
                else:
                    action()

    def arg_parser(self, command):

        args = self.prompts.get(command)

        if args is not None:
            args = dict(args)
            for key in args.keys():
                default = args[key][1]
                if args[key][0] == 'continue':
                    args[key] = default
                else:
                    args[key] = input(args[key][0])
                    if not args[key]:
                        args[key] = default
        return args
