import ipywidgets as widgets
import sys
from IPython.display import display
from IPython.display import clear_output

def create_multipleChoice_widget(description, options, correct_answer):
    if correct_answer not in options:
        options.append(correct_answer)
    
    correct_answer_index = options.index(correct_answer)
    
    radio_options = [(words, i) for i, words in enumerate(options)]
    alternative = widgets.RadioButtons(
        options = radio_options,
        description = '',
        disabled = False
    )
    
    description_out = widgets.Output()
    with description_out:
        print(description)
        
    feedback_out = widgets.Output()

    def check_selection(b):
        a = int(alternative.value)
        if a==correct_answer_index:
            s = '\x1b[1;30;42m' + "Correct" + '\x1b[0m' +"\n" #green color
        else:
            s = '\x1b[1;30;41m' + "Incorrect" + '\x1b[0m' +"\n" #red color
        with feedback_out:
            clear_output()
            print(s)
        return
    
    check = widgets.Button(description="submit")
    check.on_click(check_selection)
    
    
    return widgets.VBox([description_out, alternative, check, feedback_out])
    

Q1 = create_multipleChoice_widget('blablabla',['apple','banana','pear'],'pear')
Q2 = create_multipleChoice_widget('lalalalal',['cat','dog','mouse'],'dog')
Q3 = create_multipleChoice_widget('jajajajaj',['blue','white','red'],'white')

display(Q1)
display(Q2)
display(Q3)

