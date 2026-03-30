# Readme
This program simulates the behaviour of a flock of birds modeled using Reynolds', Vicsek's and Couzin's models. This is documented in the .pdf file in this repository. 
## Running the code
In order to run the code, one must first install the correct libraries. We suggest to do so in a different environment than usual, with python installed. In order to install the libraries, type the following on ubuntu:
'''console
sudo apt update
sudo apt install ffmpeg python3-pip -y
pip3 install numpy pandas matplotlib numba
'''
Once this is done, to run the code just type:
'''console
python main.py
'''
In order to get the desired outcome, one must modify the config.py file accordingly to their intentions. If desired, an .mp4 file will appear in the folder of the readers' choice, where the boids will be animated.

## Data and Gifs
The collected data and the gifs are inside the data folder, and can be seen by the reader. The reader might also produce new ones themselves by using the program.
