Introduction:

Unattended luggage in public spaces, such as airports and
train stations, poses significant security risks. It is a major
security concern due to its potential to harbor dangerous
materials, and its presence often leads to evacuations and
service disruptions. Unattended luggage, if left unchecked,
can trigger public panic and economic losses for nearby
businesses.

Description of the Code Files:

main.py:
How it Runs: This script is run via the command line, where the user specifies the video file, YOLO model, device (CPU or GPU), confidence threshold, and proximity threshold. It processes the video frame by frame, detecting people and luggage, and flags unattended luggage based on proximity logic. The video processing can be stopped with the "q" key.

interface.py:
How it Runs: This script provides a GUI using Tkinter, allowing users to select a video file and configure settings (model path, device, detection confidence, and threshold). It runs main.py in the background and processes videos in separate threads for convenience.

General Functionality
main.py detects people and luggage in a video using YOLOv8, determining if any luggage is abandoned based on proximity to people.

interface.py provides a user-friendly interface for selecting videos and running the detection process without using the command line.

Videos description:

• Video1 DS: This video shows a person arriving with
a suitcase, leaving it behind, and walking away. Later,
two people passed by the suitcase. After some time, the
owner returned to collect it.

• Video2 DS: The video shows three people: one with a
suitcase, another with a backpack, and the third with a
handbag. The person with the suitcase and the person
with the backpack left their luggage and walked away,
while the person with the handbag just passed by their
unattended luggage. Later, the owners who left their
luggage returned to retrieve it.

• Video3 DS: The video shows a person with a suitcase
arriving, leaving the suitcase, and walking away. Then,
another person with a backpack left his backpack and
walked away. Later, the suitcase’s owner returned to
retrieve it, followed by the backpack’s owner, who
came back to retrieve his backpack.

• Video4 DS: The video shows a person with a handbag dropping it and walking away. After a while, he
returned to retrieve it.
