Description:
Take any custom framework/pretrained model (choose StarSpace by default, look how to install it in StarSpace#requirements). Write a Dockerfile that would install it on top of the base Linux image.
Choose some dataset for your framework / model with already existing solutions (e.g. finished Kaggle competition, some tutorial) and run/execute it with a Dockerfile. 

If you chose Starspace, the Dockerfile must be able to produce embeddings for an input file via 
./starspace train -trainFile {input_file} - model modelSaveFile
where the input_file should be mounted to the container via Volume, a generated output_file should be visible and editable on the host machine (to avoid permission deny error read the article about Permissions from Materials section).
You can download and use this text file as an input file for the Starspace, or create it by yourself.
 
Organize your artifacts that would allow others to work with it.
Double-check that everything works, add commands with descriptions/comments which you used into README.md and commit to the repository.

If you choose Starspace, the structure should look like:

repository_name/containerization_hw/
|
|_ Dockerfile
|
|_ volume/ (a folder, put here an input_file for the Starspace + an output file must appear here after running the dockerfile)
|
|_ Readme.md (write here commands and comments for this task to reproduce your steps)


Criteria:
Follow best practices in using Docker:
If a service can run without privileges, use a non-root user
Use variables/arguments to avoid hard coding
Use Volume for any mutable and/or user-serviceable parts of your image
Correctly create permissions on mounted data via volume
