import os
import matplotlib.pyplot as plt

# Folder path where the files are located
folder_path = "./results/textoutput"

# Name of the output file
output_file = "combined_file2txt"

# Get a sorted list of all file names in the folder
file_names = sorted(os.listdir(folder_path), key=lambda x: int(x.split(".")[0]))

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Loop through all files in the folder
    for file_name in file_names:
        # Get the full file path
        file_path = os.path.join(folder_path, file_name)
        # Check if it is a file and not a directory
        if os.path.isfile(file_path):
            # Open the file in read mode
            with open(file_path, 'r') as infile:
                # Write the file name to the output file
                outfile.write(file_name.split(".")[0])
                outfile.write('\n')
                # Read the content of the file
                content = infile.read().strip()
                # Split the content into sentences using the comma as the delimiter
                sentences = content.split(",")
                # Loop through the sentences and write each sentence to a new line in the output file
                for sentence in sentences:
                    outfile.write(sentence.strip())
                    outfile.write('\n')
                outfile.write('\n')

# Create empty lists to store the data
labels = []
errors = []
values = []
poses = []

# Open the text file and read the contents
with open(output_file, 'r') as f:
    lines = f.readlines()

    # Loop through each line in the file
    for i in range(len(lines)):
        # Check if the line contains percentage error information
        if 'percentage error is:' in lines[i]:
            # Extract the label, pose type, and percentage error from the lines
            label = lines[i-2].strip()
            pose = lines[i-1].strip()
            percentage_error = float(lines[i].split(':')[1].strip('%\n'))

            # Add the label, pose type, and percentage error to the lists
            labels.append(label)
            poses.append(pose)
            errors.append(percentage_error)

            # Extract the value from the next line
            value = int(lines[i+1].strip())
            values.append(value)

# # Create a bar chart of the percentage errors
plt.bar(labels, errors)
plt.title('Percentage Errors')
plt.xlabel('Image Label')
plt.ylabel('Percentage Error')
plt.savefig('./results/output_plot/percentage_errors.png')
plt.show()

# Create a line chart of the values
plt.plot(labels, values)
plt.title('Angles')
plt.xlabel('Image Label')
plt.ylabel('angles')
plt.savefig('./results/output_plot/values.png')
plt.show()

# Create a histogram of the poses
plt.hist(poses)
plt.title('Poses')
plt.xlabel('Pose')
plt.ylabel('Count')
plt.savefig('./results/output_plot/poses.png')
plt.show()
