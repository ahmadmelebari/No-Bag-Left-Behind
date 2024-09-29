Description of the Code Files:

main.py:

How it Runs: Open the project in Visual Studio and run main.py from the terminal or by configuring it as the startup file. You can specify the video file, YOLO model, device (CPU or GPU), confidence, and proximity thresholds as command-line arguments. The script processes the video frame by frame, detecting people and luggage, and flags unattended luggage. Press "q" to stop the video processing.

interface.py:

How it Runs: Open and run interface.py in Visual Studio. The Tkinter-based GUI will launch, allowing you to select a video file and configure the model, device, confidence, and proximity threshold. Upon clicking "Run," the GUI will execute main.py in the background, processing the video.

General Functionality

main.py: detects people and luggage in a video using YOLOv8, determining if any luggage is abandoned based on proximity to people.

interface.py: provides a user-friendly interface for selecting videos and running the detection process without using the command line.

Videos description:

• Video1 DS: This video shows a person arriving with a suitcase, leaving it behind, and walking away. Later, two people passed by the suitcase. After some time, the owner returned to collect it.

• Video2 DS: The video shows three people: one with a suitcase, another with a backpack, and the third with a handbag. The person with the suitcase and the person with the backpack left their luggage and walked away, while the person with the handbag just passed by their unattended luggage. Later, the owners who left their luggage returned to retrieve it.

• Video3 DS: The video shows a person with a suitcase arriving, leaving the suitcase, and walking away. Then, another person with a backpack left his backpack and walked away. Later, the suitcase’s owner returned to retrieve it, followed by the backpack’s owner, who came back to retrieve his backpack.

• Video4 DS: The video shows a person with a handbag dropping it and walking away. After a while, he returned to retrieve it.
