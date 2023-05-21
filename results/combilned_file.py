import os

# Folder path where the files are located
folder_path = "textoutput"

# Name of the output file
output_file = "combined_file.txt"

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
