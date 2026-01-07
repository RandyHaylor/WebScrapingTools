import chardet
import os
import re

# List to keep track of failed files
failed_files = []

# Regex for non-standard characters (non-ASCII range)
non_standard_char = re.compile(r'[^\x00-\x7F]')

# List all txt files in the current directory (non-recursive)
for file_name in os.listdir('.'):
    if file_name.endswith('.txt'):
        # Detect encoding of the file before conversion
        f = open(file_name, 'rb')
        raw_data = f.read()
        f.close()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"Processing: {file_name}, Original Encoding: {encoding}")

        # Attempt to decode the file and check for non-standard characters
        try:
            f = open(file_name, 'r', encoding=encoding, errors='replace')
            content = f.read()
            f.close()

            # Check for non-standard characters
            if non_standard_char.search(content):
                print(f"Non-standard characters found in: {file_name}")
                failed_files.append(file_name)

                # Create a backup of the original file
                backup_name = f"{file_name}.old.txt"
                os.rename(file_name, backup_name)

                # Remove non-standard characters
                clean_content = non_standard_char.sub('', content)

                # Save the cleaned content back to the original file name
                f = open(file_name, 'w', encoding='utf-8')
                f.write(clean_content)
                f.close()
                print(f"Cleaned non-standard characters and saved backup as {backup_name}")

            else:
                # Save the content in UTF-8 encoding
                f = open(f'{file_name}.utf8', 'w', encoding='utf-8')
                f.write(content)
                f.close()

            print(f"Processed: {file_name}, Saved as UTF-8")
        except UnicodeDecodeError as e:
            print(f"Failed to decode {file_name}: {e}")
            failed_files.append(file_name)

# Print the failed files and their count at the end
if failed_files:
    print("\nThe following files have non-standard characters or failed:")
    for failed_file in failed_files:
        print(failed_file)
    print(f"\nTotal problematic files: {len(failed_files)}")
else:
    print("\nAll files processed successfully and are free of non-standard characters.")

