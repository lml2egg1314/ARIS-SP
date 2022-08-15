% use_dcts = [7,15];
warning off;
% use_dcts = [9,15,21,13,26,31];
use_dcts = {'E345', 'E45', 'E456', 'E4567', 'E45678', 'E56', 'E567', 'E5678', 'E67', 'E678', 'E78'};
for i = 1
    params.start = 1;
    for j = 1:length(use_dcts)
%         payload = 0.02*j;
        payload = 0.10;
%         if j == 5 && i==1
%             continue;
%         end
        QF = 55+10*i;
%         QF = 65;
        image_size = 256;
        params.use_dcts = use_dcts{j};
        
%         if i == 1 && j ==1
%             base_dir = '/data1/lml/watermarking';
%             image_set = 'BossBase-1.01';
%             cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
%           
%             stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
%         else
            tic;
            if i == 3
                image_set = 'BB';
                base_dir = '/data/lml/watermarking';
                cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
                stego_dir = sprintf('%s-dmmr-payload-%.2f-use-dcts-%s', cover_dir, payload, params.use_dcts); 
            else
                [cover_dir, stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload, params);
            end
            toc;
            
%         end
        params.payload = payload;
        params.quality = QF;
        params.cover_dir = cover_dir;
        params.stego_dir = stego_dir;
        
        params.listNum = 1;
        disp(params);
        dctr_and_ensemble_from_image1;
        params.start = 1;
    end
end

function [cover_dir,stego_dir] = Test_Robustness_DMMR_JUNIWARD_T(QF, image_size, payload, params)
%% ����ѹ���޸ĵĲ��Դ���
%     clear all;
%     clc;
%%  ��������
    addpath(fullfile('jpegtbx'));
    addpath(fullfile('stc_matlab'));
    para_start = params.start;
    use_dcts = params.use_dcts;
    base_dir = '/data1/lml/watermarking';
%     image_set = 'BossBase-1.01';
    image_set = 'BB';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
    recompress_dir=sprintf('%s/DMMR/recompress', base_dir); 
    if ~exist(recompress_dir,'dir')
        mkdir(recompress_dir); 
    end
    cost_dir = sprintf('%s-dmmr-cost', cover_dir); 
    if ~exist(cost_dir,'dir')
        mkdir(cost_dir); 
    end  
    stego_dir = sprintf('%s-dmmr-upward-payload-%.2f-use-dcts-%s', cover_dir, payload, params.use_dcts)
    if ~exist(stego_dir,'dir')
        mkdir(stego_dir); 
    end  
    msg_dir = sprintf('%s-dmmr-payload-%.2f-msg', cover_dir, payload); 
    if ~exist(msg_dir,'dir')
        mkdir(msg_dir); 
    end  
    afterchannel_stego_dir = sprintf('%s/DMMR/re_stego', base_dir); 
    if ~exist(afterchannel_stego_dir,'dir')
        mkdir(afterchannel_stego_dir); 
    end  
    if QF == 65
        attack_QFs = [65,70,75,80,85,90,95];
    else
        attack_QFs = [75,80,85,90,95];
    end
    attack_times = length(attack_QFs);
    cover_num = 10000; 
    cover_QF = QF; 
%     attack_QF = QF; 
%     payload = 0.1; 
    bit_error_rate = zeros(attack_times,cover_num); 
%%  ��ϢǶ��    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        recompress_Path=fullfile([ recompress_dir,'/',index_str,'.jpg']);
        stego_Path = fullfile([stego_dir,'/',index_str,'.jpg']);   
        msg_Path = fullfile([msg_dir,'/',num2str(i_img),'.mat']);
        cost_path = fullfile([cost_dir,'/',num2str(i_img),'.mat']);
        afterchannel_stego_Path = fullfile([afterchannel_stego_dir,'/',index_str,'.jpg']);   
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};  
        C_QUANT = C_STRUCT.quant_tables{1}; %����ͼ��������
        nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
%  ��������ȷֲ��Ķ�����ԭʼ�������?
        raw_msg_len = ceil(payload*nzAC);
%         if raw_msg_len<=1 %����BOSSbase��ݼ��д��ڲ��ʺ�Ƕ��Ĳ�����Ƕ��
%             imwrite(imread(cover_path),recompress_Path,'quality',cover_QF);
%             imwrite(imread(recompress_Path),stego_Path,'quality',cover_QF);
%             imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);
%             bit_error_rate(1,i_img)=0;
%             continue;
%         end
        raw_msg = round(rand(1,raw_msg_len) ); %ԭʼ������Ϣ��������
        
        n = 255; k = 251; m =8 ;    % RS�������?? 
       
        if para_start == 0
            [rho1_P,rho1_M] = J_UNIWARD_D(cover_path,1);
            save_cost(cost_path, rho1_P, rho1_M);
        else
            [rho1_P,rho1_M] = load_cost(cost_path);
        end
        [cover_lsb, change, rho] = DMMR(use_dcts,cover_path, rho1_P, rho1_M, C_QUANT);%����ͨ����Ƽ��������У��޸ľ��룬�޸Ĵ��
        %  ����STC������ϢǶ��  
        [stc_n_msg_bits,stc_n_msg_bits_check,l_check_check] = stc_embedding(use_dcts,raw_msg, cover_path, cover_lsb, rho, change,cover_QF, stego_Path,n,k,m,recompress_Path,C_QUANT);
%%  ģ���ŵ�ѹ�� 
%         save_msg(msg_Path, raw_msg, stc_n_msg_bits,stc_n_msg_bits_check,l_check_check);
        for a_t = 1:attack_times
            attack_QF = attack_QFs(a_t);
%             imwrite(imread(stego_path),afterchannel_stego_path,'quality',Facebook_attack_QF);  
            imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);    

    %%  ��Ϣ��ȡ
            [stc_decoded_msg] = stc_extracting(use_dcts,afterchannel_stego_Path, stc_n_msg_bits,stc_n_msg_bits_check, C_QUANT,n,k,m,l_check_check);   

    %%  ����ÿ��ͼ���������??      
            bit_error = double(raw_msg) - double(stc_decoded_msg);
            bit_error_number = sum(abs(bit_error))/raw_msg_len;
            bit_error_rate(a_t,i_img) = bit_error_number;
        end
        
%% ����ÿ�ų�����
%  ���ÿ��ͼ���������               
%         fprintf('%s\n',['payload: ',num2str(payload),'  image_number: ',index_str,'  error_rate: ',num2str(bit_error_rate(1,i_img))]);  %
    end
%  �������ͼ���ƽ��������
    ave_error_rates = mean(bit_error_rate, 2);
    file_id = fopen('ddmr_log_ber_new_parameter.txt', 'a');
    fprintf(file_id, '%s \n', stego_dir);
    for p_t = 1:attack_times
        attack_QF = attack_QFs(p_t);
        ave_error_rate = ave_error_rates(p_t);
    
        fprintf(file_id,'image_size-%d-QF-%d-%s\n',image_size, attack_QF, ['payload: ',num2str(payload),'  ave_error_rate: ',num2str(ave_error_rate)]);  
    end
    fclose(file_id);
end
function save_msg(msg_path, msg, stc_msg, stc_c1, stc_c2)
save(msg_path,'msg', 'stc_msg', 'stc_c1', 'stc_c2');
end

function save_cost(cost_path, rho1_P, rho1_M)
save(cost_path, 'rho1_P', 'rho1_M');
end
function [rho1_P, rho1_M] = load_cost(cost_path)
cost_mat = load(cost_path);
rho1_P = cost_mat.rho1_P;
rho1_M = cost_mat.rho1_M;
end
