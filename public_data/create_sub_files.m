clear all
original_folder = 'C:\Users\amar\Downloads\RTE_starting_kit\public_data\datasets\chronics\';
sub_sample = 7;
dest_folder = strcat('C:\Users\amar\Downloads\RTE_starting_kit\public_data\datasets_sub_',num2str(sub_sample),'\chronics');
mkdir(dest_folder)
file_id = {'_N_datetimes','_N_loads_p','_N_loads_p_planned','_N_loads_q','_N_loads_q_planned','_N_prods_p','_N_prods_p_planned','_N_prods_v','_N_prods_v_planned','hazards','maintenance','_N_simu_ids' };
for k = 0:49 %0
    chronic_num = num2str(k,'%04.f');
    for n = 1 :12
        file_name = strcat(original_folder,chronic_num,'\',file_id{n},'.csv');
        dest_file_name = strcat(dest_folder,'\',chronic_num,'\',file_id{n},'.csv');
        %         header = dlmread(filename,',',0,0,1);
        mkdir(strcat(original_folder,'test\',chronic_num))
        fid = fopen(file_name);
        header = textscan(fid,'%s',1);
        if n==1
            data = textscan(fid,'%s','Headerlines',1);
            data = cell2mat(data{1});
            %             data = cellstr(data).';
        else
            if (n == 2 && k == 12)
                data = dlmread(file_name,',',1);
            else
                data = dlmread(file_name,';',1);
            end
        end
        fclose(fid);
        fileID = fopen(dest_file_name,'w');
        textHeader = cell2mat(header{1});
        fprintf(fileID,'%s\n',textHeader);
        fclose(fileID);
        if n ==1
            dlmwrite(dest_file_name,data(1:sub_sample:end,:),'-append','delimiter','');
        elseif n==12
            dlmwrite(dest_file_name,[0:(size(data,1)-1)/sub_sample]','-append');
        else
            dlmwrite(dest_file_name,data(1:sub_sample:end,:),'-append','delimiter',';');
        end
    end
end