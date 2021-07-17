import sys
import os
import re

def clean_line(line):    
    line = line.replace(' .', '.')
    line = re.sub(r'((\.){2,}( ){1,})', r'\1\n', line)
    line = re.sub(r'\n{2,}', r'\n', line)
    return line

def main(file_path):

    file_name = os.path.basename(file_path)
    batch_file = open(file_path)

    clausified_batch_file = open('./clausified_' + file_name, 'w+')
    #lost_files = open('./lost_files.txt', 'a+')

    last_doc_id = -1
    while True:
        # Get next line from file
        line = batch_file.readline().rstrip()

        line = clean_line(line)

        line = line.replace('EDU_BREAK ', '\n')
        line = line.replace('EDU_BREAK.', '\n')

        if 'DOC_BREAK' in line:
            doc_id = int(line.split()[1].split('.')[0])
            clausified_batch_file.write('DOC_BREAK ' + line + '\n')
            continue
            # if last_doc_id == -1:
            #     last_doc_id = doc_id
            # else:
        
            #     # if (doc_id - last_doc_id) != 1:
            #     #     for doc in range(last_doc_id + 1, doc_id):
            #     #         lost_files.write(str(doc) + '\n')
                
            #     last_doc_id = doc_id
            # print(doc_id)
        clausified_batch_file.write(line + '\n')

        # if line is empty
        # end of file is reached
        if not line:
            break

    clausified_batch_file.seek(0)
        
    with open('./clausified_cleaned_' + file_name, 'w') as filehandle:
        for line in clausified_batch_file:
            if line.rstrip():
                if line.strip() != '\n' or line.strip() != '>' or line.strip() != '' or line.strip() != ',' or line.strip() != '.':
                    filehandle.write(line)
        filehandle.write('END_OF_BATCH')

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('ERROR: Provide file path! \nUsage: python clausify_batch.py [FILE_PATH]')
        sys.exit(2)

    file_path = sys.argv[1]
    main(file_path)


