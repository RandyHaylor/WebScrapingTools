import importlib
import subprocess
import sys
import os
import re


# def check_and_offer_install(packages):
#     for package in packages:
#         try:
#             importlib.import_module('bs4')
#             print("beautifulsoup4 is installed")
#         except ImportError:
#             print(f"Required module '{package}' is not installed.")
#             install = input(f"Do you want to install '{package}' now? (yes/no): ").strip().lower()
#             if install == 'yes':
#                 try:
#                     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#                     print(f"Module '{package}' installed successfully.")
                    
#                 except subprocess.CalledProcessError as e:
#                     print(f"Failed to install '{package}'. Error: {e}")
#                     sys.exit(1)
#             else:
#                 print(f"Module '{package}' was not installed. Exiting program.")
#                 sys.exit(1)

# # 
# requirements = ['beautifulsoup4', 'concurrent.futures', 'functools', 'multiprocessing']
# check_and_offer_install(requirements)

from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# Config vars
target_extensions = ["html", "htm", "xhtml"]  # list of extensions to include
output_folder_name = "extractedtext"
target_tag_names = ["div"]
target_classes = ["section", "subsection"]
omit_classes = ["breadcrumbs clear", "nextprev clear", "footer-wrapper", "sidebar", "header-wrapper"]
output_extension = ".txt"
overwrite = True
## later, the cwd will be changed to identify a source location, this script uses current working directory
## the output is cwd\output_folder_name - so again, update the cwd before ineracting with output folder
##    to make it other than contained in the target cwd


def GetContentInTagWithSpecificClass(html_content, target_tags, target_classes, remove_classes):
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove all elements with classes in remove_classes
    for remove_class in remove_classes:
        for elem in soup.find_all(class_=remove_class):
            elem.decompose()
            
    # Find all remaining matching tags with specified class
    elements = soup.find_all(target_tags, class_=target_classes)

    output = ""

    # Extract and process text
    for element in elements:        
        # Handle <br>, <br/>, and <p> tags by replacing them with newlines
        for br in element.find_all("br"):
            br.replace_with("\n")
        for p in element.find_all("p"):
            p.append("\n")
            p.unwrap()

        # Remove HTML tags and get clean text
        element_text = element.get_text()

        # Append cleaned text to output with two line breaks
        output += element_text + "\n\n"

    return output


def GetListOfFilesAndPathsByExtension(extensions):
    file_list = []
    base_path = os.getcwd()
    # Normalize the extensions to ensure matching is case insensitive and consistent
    extensions = [ext.lower() for ext in extensions]

    # Walk through all directories and files starting from base_path
    for dirpath, _, filenames in os.walk(base_path):
        for file in filenames:
            # Check if the file extension is in the list of extensions we're interested in
            if any(file.lower().endswith(ext) for ext in extensions):
                # Construct the full path to the file
                full_path = os.path.join(dirpath, file)
                # Subtract the base_path and leading separator to fit the requirement
                relative_path = os.path.relpath(full_path, start=base_path)
                file_list.append(relative_path)
                print(f"File     found: {relative_path}")

    return file_list



def ProcessFilePath(absolute_output_path, absolute_source_path, filepath, tag_names, tag_classes, classes_to_ignore, output_extension, overwrite):
    absolute_output_file_path = os.path.join(absolute_output_path, filepath) + output_extension
    absolute_source_file_path = os.path.join(absolute_source_path, filepath)        
    print(f"Processing: {absolute_source_file_path}")
    
    if os.path.isfile(absolute_output_file_path) and not overwrite:  # Check if file exists and overwrite is False
        print("Skipping existing file (overwrite disabled):", absolute_source_file_path)
        #files_skipped += 1
        return

    os.makedirs(os.path.dirname(absolute_output_file_path), exist_ok=True)  # Ensure directory exists
    
    with open(absolute_source_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    file_name = os.path.basename(filepath)

    extracted_text = file_name + GetContentInTagWithSpecificClass(html_content, tag_names, tag_classes, classes_to_ignore)

    #reduce any consecutive line break strings down to 2 line breaks
    # This regex matches three or more newline characters and replaces them with exactly two newlines
    cleand_extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text)

    with open(absolute_output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(cleand_extracted_text)
        print("File created:", absolute_output_file_path)        
    
    return "Processed"



def ExtractTextInsideTargetTags(file_extensions, tag_names, tag_classes, output_extension, output_folder_name, overwrite, classes_to_ignore):
    absolute_source_path = os.getcwd()
    absolute_output_path = os.path.join(absolute_source_path, output_folder_name)
    # attempt to create output dir
    os.makedirs(output_folder_name, exist_ok=True)

    filepaths_to_process = GetListOfFilesAndPathsByExtension(file_extensions)
    print(f"Files Found: {len(filepaths_to_process)}")
        
    process_function = partial(ProcessFilePath, absolute_output_path, absolute_source_path, tag_names=tag_names, tag_classes=tag_classes, classes_to_ignore=classes_to_ignore, output_extension=output_extension, overwrite=overwrite)

   # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_function, filepaths_to_process))
        
    print(f"Completed attempting for {len(filepaths_to_process)} files found.")

## main program
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    ExtractTextInsideTargetTags(target_extensions, target_tag_names, target_classes, output_extension, output_folder_name, overwrite, omit_classes)

# testing extraction: test.html
# def test():
#     with open("test.html", 'r', encoding='utf-8') as file:
#         html_content = file.read()
        
#     print(html_content)
    
#     output = GetContentInTagWithSpecificClass(html_content, "div", "subsection")
    
#     print(f"output: test.html: {output}")

# test()
