import Tkinter, Tkconstants, tkFileDialog, tkSimpleDialog, tkMessageBox
import os
import shutil
import thread
import pyaudio
import speakerclassifier
import wave
from tkintertable import TableCanvas, TableModel
import time

class SpeakerRecognizerFrame(Tkinter.Frame):

  # Initializes the Main Application Frame
  def __init__(self, root):
	
    Tkinter.Frame.__init__(self, root)
    
    # default_mode
    self.root = root
    self.state = 'normal'

    # options for buttons
    button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}

	  # define labels
    Tkinter.Label(self, text="Current Audio", font=("Helvetica", 12)).grid(row=0, column=0, columnspan = 6)
    self.label = Tkinter.Label(self, text="None", font=("Helvetica", 12), width=50, fg="blue")
    self.label.grid(row=1, column=0, columnspan = 6)
	
    # define buttons
    self.button_record = Tkinter.Button(self, text='Record', command=self.record_finish_audio)
    self.button_record.grid(padx=5, pady=10, row=3, column=1)  
    self.button_play = Tkinter.Button(self, text='Play', command=self.play_stop_audio, state='disable')
    self.button_play.grid(padx=5, pady=10, row=3, column=2)
    self.button_load = Tkinter.Button(self, text='Load File', command=self.askopenfilename)
    self.button_load.grid(padx=5, pady=10, row=3, column=3)
    self.button_save = Tkinter.Button(self, text='Save As', command=self.asksaveasfilename, state='disable')
    self.button_save.grid(padx=5, pady=10, row=3, column=4)
    self.button_train = Tkinter.Button(self, text='Train', command=self.train_handler, state='disable')
    self.button_train.grid(padx=5, pady=10, row=4, column=2)
    self.button_classify = Tkinter.Button(self, text='Classify', command=self.classify_handler, state='disable')
    self.button_classify.grid(padx=5, pady=10, row=4, column=3)

    # define options for opening or saving a file
    self.file_opt = options = {}
    options['defaultextension'] = '.wav'
    options['filetypes'] = [('wave files', '.wav')]
    options['parent'] = root
    options['title'] = 'Load / Save Wave file'

    # This is only available on the Macintosh, and only when Navigation Services are installed.
    #options['message'] = 'message'

    # if you use the multiple file version of the module functions this option is set automatically.
    #options['multiple'] = 1

    # defining options for opening a directory
    self.dir_opt = options = {}
    #options['initialdir'] = 'C:\\'
    options['mustexist'] = False
    options['parent'] = root
    options['title'] = 'Hello '

    
  # Captures audio input from microphone
  def record_audio(self, dummy):
    self.label['text'] = 'Recording...'
    
    # Set audio file settings
    p = pyaudio.PyAudio()
    stream = p.open(format=speakerclassifier.FORMAT,
                channels=speakerclassifier.CHANNELS,
                rate=speakerclassifier.SAMPLE_RATE,
                input=True,
                frames_per_buffer=speakerclassifier.CHUNK) #buffer
    frames = []
     
    # Record
    while self.state == 'recording' :
      data = stream.read(speakerclassifier.CHUNK)
      frames.append(data)
    
    # Close Stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write to wave file
    wf = wave.open(os.path.join(os.getcwd(), speakerclassifier.RECORDING_TEMPFILENAME), 'wb')
    wf.setnchannels(speakerclassifier.CHANNELS)
    wf.setsampwidth(p.get_sample_size(speakerclassifier.FORMAT))
    wf.setframerate(speakerclassifier.SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Enable GUI components
    self.button_play['state'] = 'normal'
    self.button_load['state'] = 'normal'
    self.button_save['state'] = 'normal'
    self.button_classify['state'] = 'normal'
    self.button_train['state'] = 'normal'
    self.button_record['text'] = 'Record'
    self.label['text'] = speakerclassifier.RECORDING_TEMPFILENAME
    self.currentFile = os.path.join(os.getcwd(), speakerclassifier.RECORDING_TEMPFILENAME)
     
  # Record Button Click Event Handler
  def record_finish_audio(self):
  
    # Start Recording
    if self.state != 'recording' :
      self.state = 'recording'
      self.button_play['state'] = 'disable'
      self.button_load['state'] = 'disable'
      self.button_save['state'] = 'disable'
      self.button_classify['state'] = 'disable'
      self.button_train['state'] = 'disable'
      self.button_record['text'] = 'Finish'
      thread.start_new_thread(self.record_audio, (self,) )
      
    # Stop Recording
    else :
      self.state = 'normal'

  # Plays the current Audio Clip
  def play_audio(self, dummy):
    self.label['text'] = 'Playing ' +  self.label['text'];
    
    # open audio file
    f = wave.open(self.currentFile,"rb")
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(speakerclassifier.CHUNK)

    # play audio file
    while data != '' and self.state == 'playing':  
      stream.write(data)  
      data = f.readframes(speakerclassifier.CHUNK)  
    
    # close streams  
    stream.stop_stream()  
    stream.close()    
    p.terminate()  
    
    # Enable GUI components
    self.state = 'normal'
    self.button_record['state'] = 'normal'
    self.button_load['state'] = 'normal'
    self.button_save['state'] = 'normal'
    self.button_classify['state'] = 'normal'
    self.button_train['state'] = 'normal'
    self.button_play['text'] = 'Play'
    self.label['text'] = os.path.basename(self.currentFile)
 
  # Play Button Click Event Handler 
  def play_stop_audio(self):
    if self.state != 'playing' :
      self.state = 'playing'
      self.button_record['state'] = 'disable'
      self.button_load['state'] = 'disable'
      self.button_save['state'] = 'disable'
      self.button_classify['state'] = 'disable'
      self.button_train['state'] = 'disable'
      self.button_play['text'] = 'Stop'
      thread.start_new_thread(self.play_audio, (self,) )
    
    # Stop Playing Audio
    else :
      self.state = 'normal'
   

  # Loads a saved wave file
  def askopenfilename(self):
    # get filename
    filename = tkFileDialog.askopenfilename(**self.file_opt)

    # open file on your own
    if filename:
      self.label['text'] = os.path.basename(filename)
      
      self.button_play['state'] = 'normal'
      self.button_save['state'] = 'normal'
      self.button_train['state'] = 'normal'
      self.button_classify['state'] = 'normal'
       
      self.currentFile = filename

   # Saves the current wave file/recording
  def asksaveasfilename(self):
    # get filename
    destFilename = tkFileDialog.asksaveasfilename(**self.file_opt)

    # open file on your own
    if destFilename:
      try :
        shutil.copy(self.currentFile, os.path.dirname(destFilename))
      except :
        print ''
      os.rename(os.path.join(os.path.dirname(destFilename), os.path.basename(self.currentFile)),   os.path.join(os.path.dirname(destFilename),os.path.basename(destFilename)))
      self.label['text'] = os.path.basename(destFilename)
      self.currentFile = destFilename
      
      print self.currentFile

  # Train Button Click Event Handler
  def train_handler(self):

    name = tkSimpleDialog.askstring('Train', 'Training the system with current audio file. \n\nWhat is your name?')
    if name is None :
      tkMessageBox.showerror('Train', 'You must enter a name to associate the current audio file.')
    else :
      # Show Training waiting window
      wdw = Tkinter.Toplevel()
      wdw.title('Training')
      Tkinter.Label(wdw, text="SVM Training in Progress... Please wait", font=("Helvetica", 12), width=50, fg="blue").pack()
      wdw.update()
      wdw.deiconify()
      speakerclassifier.train_user_svms(name, self.currentFile)
      wdw.destroy()
      tkMessageBox.showinfo('Train', 'Training complete')
      
      
      
  
  # Classify Button Click Event Handler
  def classify_handler(self):

    # Show Classification waiting window
    wdw = Tkinter.Toplevel()
    wdw.title('Classification Results')
    Tkinter.Label(wdw, text="Classification in Progress... Please wait", font=("Helvetica", 12), width=50, fg="blue").pack()
    wdw.update()
    wdw.deiconify()
  
    # Predict and load results
    resultModel = TableModel()
    resultDict = speakerclassifier.classify_audio(self.currentFile)
    if len(resultDict) > 0 :
      resultModel.importDict(resultDict)
    wdw.destroy()
  
    if len(resultDict) > 0 :
      # Show Classification results in modal table window
      wdw = Tkinter.Toplevel()
      wdw.geometry('300x200+200+200')
      wdw.title('Classification Results')
      tframe = Tkinter.Frame(wdw)
      tframe.pack()
      
      table = TableCanvas(tframe, model=resultModel, editable=False)
      table.createTableFrame()
      table.sortTable(columnName='Score (%)', reverse=True)
      
      wdw.transient(self.root)
      wdw.grab_set()
      self.root.wait_window(wdw)
    else :
      tkMessageBox.showerror('Classification Results', 'There are currently no users in the System')
    
    
if __name__=='__main__':
  speakerclassifier.init()
  root = Tkinter.Tk()
  SpeakerRecognizerFrame(root).grid()
  root.title('Speaker Recognizer')
  root.geometry('450x175+0+0')
  root.mainloop()