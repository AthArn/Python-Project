INSTRUCTIONS TO RUN THE PROJECT:

1)Install Python 3.6.

2)Go to command prompt and download numpy,keras,tensorflow,matlib plot and pandas libraries using pip install command
  eg: pip install numpy

3)Using command prompt,go to the directory where the project is located.

4)Type 'python model.py' in the command prompt now.

5)Wait for atleast 10 iterations of epoch to get completed.

6)Now press Ctrl and C together. This will cause a keyboard interruption.

7)In the location of the project,there will be a folder called 'models'.

8)Open the folder and copy the filename of the file with least number after the '-'
  If the files present are model.01-0-99.hdf5, model.02-0-34.hdf5,model.03-0-15.hdf5
  Copy the name of the last file(model.03-0-15.hdf5)

9)Open predict.py with a text editor (preferably notepad++ or Atom).

10)In the 4th line,remove the pre-existing file name(if any) and add the new file name.

11)Open command prompt and type 'python predict.py'

12)There will be a new CSV file named 'my_submission.csv'. 

13)This file contains the recognised digits with their image_id.

14)Using command prompt,run plotting.py file. A graph will be displayed which plots accuracy and loss against epochs.
