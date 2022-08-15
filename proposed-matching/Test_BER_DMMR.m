use_dcts = [7,15];
for i = 1:2
    for j = 1:3
        payload = 0.05*j;
        QF = 55+10*i;
        image_size = 256*2;
%         if i == 1 && j ==1
%             base_dir = '/data1/lml/watermarking';
%             image_set = 'BossBase-1.01';
%             cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
%           
%             stego_dir = sprintf('%s-dmmr-payload-%.2f', cover_dir, payload); 
%         else
            tic;
            [cover_dir, stego_dir] = Test_Robustness_DMMR_JUNIWARD_TE(QF, image_size, payload);
            toc;
%         end
        params.payload = payload;
        params.quality = QF;
        params.cover_dir = cover_dir;
        params.stego_dir = stego_dir;
        params.start = 0;
        params.listNum = 1;
        disp(params);
%         dctr_and_ensemble_from_image;
    end
end

function [cover_dir,stego_dir] = Test_Robustness_DMMR_JUNIWARD_TE(QF, image_size, payload)
%% ����ѹ���޸ĵĲ��Դ���
%     clear all;
%     clc;
%%  ��������
    addpath(fullfile('jpegtbx'));
    addpath(fullfile('stc_matlab'));
    base_dir = '/data1/lml/watermarking';
    image_set = 'BossBase-1.01';
    cover_dir = sprintf('%s/%s-cover-resample-%d-jpeg-%d',base_dir, image_set, image_size, QF); 
    recompress_dir=sprintf('%s/DMMR/recompress', base_dir); 
    if ~exist(recompress_dir,'dir')
        mkdir(recompress_dir); 
    end 
    stego_dir = sprintf('%s-dmmr-payload-%.2f-bertest', cover_dir, payload); 
    if ~exist(stego_dir,'dir')
        mkdir(stego_dir); 
    end  
    afterchannel_stego_dir = sprintf('%s/DMMR/re_stego_test', base_dir); 
    if ~exist(afterchannel_stego_dir,'dir')
        mkdir(afterchannel_stego_dir); 
    end  
    cover_num = 1000; 
    cover_QF = QF; 
    attack_QF = QF; 
%     payload = 0.1; 
    bit_error_rate = zeros(11,cover_num); 
    
    emb_fre = zeros(8);
    for i  =1:8
        for j = 1:8
            if i+j == 5 || i+j == 6
                emb_fre(i,j) = 1;
            end
        end
    end
    rep_emb_fre = repmat(emb_fre, image_size/8, image_size/8);
    rep_emb_fre = logical(rep_emb_fre);
    
%%  ��ϢǶ��    
    parfor i_img = 1:cover_num
        index_str = sprintf('%d', i_img);
        cover_path = fullfile([cover_dir,'/',index_str,'.jpg']);  
        recompress_Path=fullfile([ recompress_dir,'/',index_str,'.jpg']);
        stego_path = fullfile([stego_dir,'/',index_str,'.jpg']);    
        afterchannel_stego_Path = fullfile([afterchannel_stego_dir,'/',index_str,'.jpg']);   
        C_STRUCT = jpeg_read(cover_path);
        C_COEFFS = C_STRUCT.coef_arrays{1};  
        C_QUANT = C_STRUCT.quant_tables{1}; %����ͼ��������
        nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
%  ��������ȷֲ��Ķ�����ԭʼ�������?
        raw_msg_len = ceil(payload*nzAC);
        if raw_msg_len<=1 %����BOSSbase��ݼ��д��ڲ��ʺ�Ƕ��Ĳ�����Ƕ��
            imwrite(imread(cover_path),recompress_Path,'quality',cover_QF);
            imwrite(imread(recompress_Path),stego_path,'quality',cover_QF);
            imwrite(imread(stego_path),afterchannel_stego_Path,'quality',attack_QF);
            bit_error_rate(:,i_img)=0;
            continue;
        end
        raw_msg = round( rand(1,raw_msg_len) ); %ԭʼ������Ϣ��������    
        n = 255; k = 251; m =8 ;    % RS�������??? 
        [rho1_P,rho1_M] = J_UNIWARD_D(cover_path,1);%ʹ��J_UNIWARD��д�㷨����DCTϵ���Ƕ����???
        [cover_lsb, change, rho] = DMMR(cover_path, rho1_P, rho1_M, C_QUANT);%����ͨ����Ƽ��������У��޸ľ��룬�޸Ĵ��
        %  ����STC������ϢǶ��  
        [stc_n_msg_bits,stc_n_msg_bits_check,l_check_check] = stc_embedding(raw_msg, cover_path, cover_lsb, rho, change,cover_QF, stego_path,n,k,m,recompress_Path,C_QUANT,attack_QF);
%%  ģ���ŵ�ѹ��
%         C_STRUCT = jpeg_read(cover_path);
%         C_COEFFS = C_STRUCT.coef_arrays{1};  
%         C_QUANT = C_STRUCT.quant_tables{1}; 
%         nzAC = nnz(C_COEFFS) - nnz(C_COEFFS(1:8:end,1:8:end));
        
        S_STRUCT = jpeg_read(stego_path);
        S_COEFFS = S_STRUCT.coef_arrays{1}; 
%         size(S_COEFFS)
        tab_m = S_STRUCT.quant_tables{1}; 
        ae_S_STRUCT = C_STRUCT;
        
        Modifications = S_COEFFS - C_COEFFS;
%         disp(sum(abs(Modifications(:))));
        tmp_ber = zeros(11,1);
        for kkkk = 1:11
            myp1 = 0.01 * (kkkk-1);
            if kkkk > 1
                
            test = sign(rand(image_size)-myp1);
            test_locations = test .* rep_emb_fre;
            test_Modifications = Modifications .* test_locations;
%             disp(sum(abs(test_Modifications(:))));
            ae_S_STRUCT.coef_arrays{1} = C_COEFFS +test_Modifications;
            jpeg_write(ae_S_STRUCT, stego_path);
            recompression(stego_path,QF,stego_recom_path,tab_m,QF);
            end
        
        imwrite(imread(stego_path),afterchannel_stego_Path,'quality',attack_QF); 
        
      
%%  ��Ϣ��ȡ
        [stc_decoded_msg] = stc_extracting(afterchannel_stego_Path, stc_n_msg_bits,stc_n_msg_bits_check, C_QUANT,n,k,m,l_check_check);   

%%  ����ÿ��ͼ���������???      
        bit_error = double(raw_msg) - double(stc_decoded_msg);
        bit_error_number = sum(abs(bit_error));
        tmp_ber(kkkk) = bit_error_number/raw_msg_len;
        end
        bit_error_rate(:,i_img) = tmp_ber
        
%% ����ÿ�ų�����
%  ���ÿ��ͼ���������               
%         fprintf('%s\n',['payload: ',num2str(payload),'  image_number: ',index_str,'  error_rate: ',num2str(bit_error_rate(1,i_img))]);  %
    end
%  �������ͼ���ƽ��������
    ave_error_rate = mean(bit_error_rate,2)
    file_id = fopen('ddmr_log.txt', 'a');
    
    fprintf(file_id,'image_size-%d-QF-%d-%s-%12.6f\n',image_size, QF, ['payload: ',num2str(payload),'  ave_error_rate: '], ave_error_rate);  
    fclose(file_id);
end
