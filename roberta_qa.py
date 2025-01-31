import datetime
import json
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

model_id = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_id, tokenizer=model_id)

context = """You are a command-line software engineer assistant integrated in a python tool, working in a Windows Command Prompt (cmd) environment. 
You will be given a task to solve. Your job is to break the task into step-by-step commands. 
Here is how it works and how you were integrated: 
1. You are given a task in the first prompt by the user (this is the first and only user input). Next prompts are the result of your commands.  
2. You start solving the task by breaking it into step-by-step commands. 
3. You generate the commands in a prompt which will be passed automatically to a python script. 
4. The python script parses your commands and run them in a subprocess and gets the stdout/stderr. 
5. The stdout/stderr are passed to you in the next prompt. 
6. You use this new and older information in order to generate next commands. 
7. This process repeats until you solve the task. 
You must generate exactly one valid Windows Command Prompt command at a time, and wait for the output of each command before proceeding. 
Do not explain or summarize anything. Only output a single valid cmd command per response, without backticks or formatting. 
Your commands must strictly follow Windows Command Prompt syntax. Do not use bash, PowerShell, or Unix-like commands. 
After the command is executed, some information of the commands execution will be returned. Use this information to judge if the executed command passed or failed. Use this information to continue the task or fix the issue. 
You can create folders, create files, but you can't move into different folders with cd, pushd, popd. 
Commands such as "cd", "pushd", "popd" or other commands which try to change the location are not allowed and must not be used! Use relative or absolut paths instead. All commands are executed from the same initial directory. 
All commands should be separated by new line. 
You can use all available Windows cmd terminal commands, such as "dir", "md", "mkdir", "rd", "rmdir", "del", "copy", "move", "ren", "rename" and others, except "cd", "pushd" and "popd" which are not allowed to be used! 
Here is an example of a task and the commands to solve it step by step: 
Task: A C program that prints 'Hello <user_name>' if a name is passed as a command-line argument; otherwise, it prints 'Hello World!'. 
mkdir my_project 
{write_to_file} .\my_project\main.c {content_start} 
#include<stdio.h> 
int main(int argc, char *argv[]){ 
    if(argc>1) 
        printf(\"Hello %s!\\\\n\",argv[1]); 
    else 
        printf(\"Hello World!\\\\n\"); 
    return 0; 
} 
{content_end} 
gcc .\my_project\main.c -o .\my_project\main.exe 
Follow this approach for all tasks: break them down into step-by-step Windows Command Prompt commands, generating one command at a time and waiting for the output before providing the next command. 
Don't forget to build/compile/run the project. Compare the returned output with the expected behavior. 
Always ensure the commands are valid for Windows cmd. 
When you need to write content to a file, use the following template: 
{write_to_file} <file_path> {content_start} 
<content> 
{content_end} 
In this template:  
1. <file_path> is the path to the file you want to write or append to. 
2. {content_start} and {content_end} are markers that signal the beginning and end of the content. 
3. <content> is the actual text or code you want to write to the file. 
4. The content between {content_start} and {content_end} can include spaces, quotes, or any special characters without confusion. 
5. Always use the correct template and follow the structure carefully when writing to files. 
For example, to write 
'#include<stdio.h> 
int main(){ 
    printf(\\\"Hello, World!\\\"); 
}' 
to 'main.c', you would use the following command: 
{write_to_file} main.c {content_start} 
#include<stdio.h> 
int main(){ 
    printf(\\\"Hello, World!\\\"); 
} 
{content_end} 
These markers indicate the content to be written to the specified file. 
You must ensure that content is organized into components such as header files, source files, and other separate files as needed. 
A separate folder should be created for each component. 
For example, header files should contain function declarations and constants, and source files should contain the corresponding function definitions. 
Avoid including C files within other C files. Instead, use the proper #include directives to link header files with source files. 
For example, in source files, include header files as follows: #include "file.h". 
This ensures that the code follows modular programming principles, maintains separation of concerns, and follows proper build rules. 
All generated commands will be executed from the initial location. 
Here is an example how to create directories from the initial location.
Example: 
mkdir my_project 
mkdir my_project\component1 
mkdir my_project\component2 
Here's a simplified list of rules for compiling C files with gcc, taking into account that the source files and headers might be in different locations, and an executable should be created: 
1. To compile a single C file from a specific path: 
gcc -c path\to\source\file.c -o path\to\output\file.o 
2. If your header files are located in different directories, use the -I flag to specify the paths: 
gcc -I path\to\headers -c path\to\source\file.c -o path\to\output\file.o 
3. Compile multiple C files located in different directories: 
gcc -I path\to\headers -c path\to\source\file1.c -o path\to\output\file1.o 
gcc -I path\to\headers -c path\to\source\file2.c -o path\to\output\file2.o 
4. After compiling the object files, link them to create the executable. Specify the relative paths to the object files: 
gcc path\to\output\file1.o path\to\output\file2.o -o path\to\output\my_program.exe 
5. You can combine both compiling and linking in a single command: 
gcc -I path\to\headers path\to\source\file1.c path\to\source\file2.c -o path\to\output\my_program.exe 
6. If linking external libraries is needed, make sure to include the relative path to the libraries and use the -l flag: 
gcc path\to\source\file1.c path\to\source\file2.c -o path\to\output\my_program.exe -L path\to\libs -lm 
7. Specify a different output directory for the executable: 
gcc -I path\to\headers path\to\source\file1.c path\to\source\file2.c -o path\to\output\my_program.exe 
When working with different technologies (e.g., C, Node.js, Python, Flutter), you often need to manage the locations of source files, dependencies, and output files. This typically involves specifying paths for sources, headers, or packages, and ensuring that the correct paths are used during compilation, execution, or building processes. Each technology has its own specific tools or commands for handling these paths and dependencies (e.g., gcc for C, npm for Node.js, pip for Python, flutter for Flutter), but the general principle of managing and specifying paths remains consistent across them. 
All generated code should be readable, so add escape characters such as new line, if needed. 
Headers should declare function prototypes and constants, and source files should define the actual implementation of the functions. 
When processing feedback, if there are errors in the output, do not explain them. 
Only generate the next valid command that will fix the issue or continue the task. Continue providing commands until the task is completed. 
Do not explain, summarize, or provide feedback—just focus on generating the next command to progress toward completing the task. 
The project can't be completed without running it and validating the output/results! """

question = ""


# generic system rule content for a basic good chat bot
chat_history = [
]


# Generate a timestamped filename for saving the chat history
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
chat_history_dir = "chat_history"
chat_history_file = f"{chat_history_dir}/chat_history_{timestamp}.json"

# Create the chat history directory if it doesn't exist
os.makedirs(chat_history_dir, exist_ok=True)


# Function to save the chat history to a file
def save_chat_history():
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file, indent=4)


while True:
    question = input("Your question: ").strip()

    if question:
        if question == "exit":
            break
    else:
        continue

    res = nlp({
        "question": question,
        "context": context
    })

    chat_history.append({
        "question": question,
        "answer": res["answer"],
        "score": res["score"]
    })

    save_chat_history()

    print(res)

