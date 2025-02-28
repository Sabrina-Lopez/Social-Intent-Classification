adults_gaze_data = dir('./adults-gaze-data/');
adults_gaze_data(1:2, :) = [];
filename = adults_gaze_data(1);

disp(filename)

disp([filename.folder, '/', filename.name])

if exist([filename.folder, '/', filename.name], 'file')
    data = readtable([filename.folder, '/', filename.name], 'FileType', 'text', 'Delimiter', '\t');
    disp('Data imported successfully:');
    disp(head(data));
else
    error('File %s does not exist in the current directory.', filename);


end