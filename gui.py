import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading

def run_script():
    output_text.delete('1.0', tk.END)

    def target():
        process = subprocess.Popen(
            ['python3', 'train.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_text.insert(tk.END, output)
                output_text.see(tk.END)
        rc = process.poll()
        output_text.insert(tk.END, 'Return Code: {}\n'.format(rc))

    threading.Thread(target=target).start()


root = tk.Tk()
root.title('GUI')
root.geometry('800x600')

output_text = ScrolledText(root, width=100, height=30)
output_text.pack()

run_button = tk.Button(root, text='Run', command=run_script)
run_button.pack()

root.mainloop()