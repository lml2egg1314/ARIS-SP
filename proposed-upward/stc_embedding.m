function [l,l_check,l_check_check, code] = stc_embedding(use_dcts,msg,coverPath,cover_lsb,rho,change,cover_QF,stegoPath,n,k,m,recompress_Path,tab_m)%�����޸Ĳ���
[min_value, max_value, block1, block2] = get_sum(use_dcts);
usable_DCT_num = sum(min_value-1:max_value-1);
%% ������ϢǶ���У����Ƕ��ķֽ��?
L=length(cover_lsb);
% block1=3;
% block2=1;
l_cover_embed=block1*k*m;%C1�ĳ���
l=length(msg);

%% ��������������
rand('seed',3); %��������
SH = randperm(1*L);
[cover_lsb_Shuffle] = Image_Shuffle(cover_lsb,SH);
[rho_Shuffle]=Image_Shuffle(rho,SH);

%% STCs ����
H = 10;
[min_cost, stc_msg] = stc_embed(uint8(cover_lsb_Shuffle(1:l_cover_embed)'),  uint8(msg'),rho_Shuffle(1:l_cover_embed), H); % Ƕ����Ϣ    
stc_extract_msg2 = stc_extract(stc_msg, l, H); % ��ȡ��Ϣ

%% ��֤STC�����Ƿ���������
% if all(uint8(msg) == stc_extract_msg2')
%     disp('Message can be extracted by STC3 correctly.');
% else
%     error('Some error occured in the extraction process of STC3.');
% end
% size(msg)

%% RS����
l_embed_check=block2*k*m;%C2�ĳ���
[Check_code]=rs_encode_klf(stc_msg,n,k,m);%RS����
l_check=length(Check_code);
[min_cost, stc_msg_check] = stc_embed(uint8(cover_lsb_Shuffle(l_cover_embed+1:l_cover_embed+l_embed_check)'),  uint8(Check_code'),rho_Shuffle(l_cover_embed+1:l_cover_embed+l_embed_check), H); % embed message    �����������ƻ���
stc_extract_check = stc_extract(stc_msg_check, l_check, H);
%% ��֤STC�����Ƿ���������
% if all(uint8(Check_code) == stc_extract_check')
%     disp('Message can be extracted by STC3 correctly.');
% else
%     error('Some error occured in the extraction process of STC3.');
% end

%% RS����
[Additional_check_code ]=rs_encode_klf(stc_msg_check,n,k,m);%����Additional check code 
l_check_check=length(Additional_check_code);
[min_cost, stc_msg_check_check] = stc_embed(uint8(cover_lsb_Shuffle(l_cover_embed+l_embed_check+1:L)'),  uint8(Additional_check_code'),rho_Shuffle(l_cover_embed+l_embed_check+1:L), H); % embed message    �����������ƻ���
stc_extract_check_check = stc_extract(stc_msg_check_check, l_check_check, H);
%% ��֤STC�����Ƿ���������
% if all(uint8(Additional_check_code) == stc_extract_check_check')
%     disp('Message can be extracted by STC3 correctly.');
% else
%     error('Some error occured in the extraction process of STC3.');  
% end
%% �������е�����
cover_lsb_Shuffle(1:l_cover_embed)=stc_msg;
cover_lsb_Shuffle(l_cover_embed+1:l_cover_embed+l_embed_check)=stc_msg_check;
cover_lsb_Shuffle(l_cover_embed+l_embed_check+1:L)=stc_msg_check_check;
[cover_lsb_change] = Image_ReShuffle(cover_lsb_Shuffle,SH);

%%  DCT�任
code = cover_lsb_change;
code_n = length(cover_lsb);
bits = 8;     
cover_spa = imread(coverPath);
cover_spa = double(cover_spa) - 2^(round(bits)-1);
[xm,xn] = size(cover_spa);
t = dctmtx(8);  %����DCT����
fun = @(xl) (t*xl*(t'));
cover_DCT = blkproc(cover_spa,[8 8],fun);%�ֿ�DCT�任
m_block = floor(xm/8);
n_block = floor(xn/8);

%% Ƕ�����?
G = 1;
n_msg = 0;
for bm = 1:m_block
    for bn = 1:n_block
        for i = 1:8
            for j = 1:8
%                 if (i+j==5)||(i+j==6)  %��Ƶ21��DCTϵ��(i+j==6)||(i+j==7)%
                if (i+j>=min_value)&&(i+j<=max_value)
                    n_msg = n_msg + 1;
                    if n_msg<=code_n
                        yd = cover_DCT((bm-1)*8+i,(bn-1)*8+j); %��������DCTϵ��
                        if code(n_msg) ~= cover_lsb(n_msg)   % ʵ���޸Ĳ���
                            yd = yd + change(n_msg); 
                        else
                            yd = cover_DCT((bm-1)*8+i,(bn-1)*8+j); %����code(n_msg) == cover_lsb(n_msg)������Ƕ��λ�ñ��ֲ���
                        end
                        cover_DCT((bm-1)*8+i,(bn-1)*8+j) = yd; 
                    else
                        break;
                    end

                end
            end
        end
        if n_msg>code_n
            break; end
    end
    if n_msg>code_n
        break; end
end


%% �޸Ľ׶� �޸Ĳ��ȶ���DCTϵ��
% for i=1:2
%     cover_spa = blkproc(cover_DCT,[8 8],'P1*x*P2',t',t);
%     cover_spa = cover_spa + double(2^(bits-1));
%     cover_spa = uint8(cover_spa);
%     imwrite(cover_spa,recompress_Path,'quality',attack_QF);
%     bits = 8;     
%     cover_spa = imread(recompress_Path);
%     cover_spa = double(cover_spa) - 2^(round(bits)-1);
%     t = dctmtx(8);  %����DCT����
%     fun = @(xl) (t*xl*(t'));
%     cover_DCT = blkproc(cover_spa,[8 8],fun);%�ֿ�DCT�任
%     n_msg = 0;
%     code_n = m_block*n_block*9;  %�е�Ƶ9��DCTϵ��
%     e_code = zeros(1,code_n);
%     for bm = 1:m_block
%         for bn = 1:n_block
%             for i = 1:8
%                 for j = 1:8
%                     if (i+j==5)||(i+j==6)  %��Ƶ9��DCTϵ�� ��ȡ��������
%                         n_msg = n_msg + 1;
%                          if n_msg<=code_n
%                             yd = cover_DCT((bm-1)*8+i,(bn-1)*8+j);
%                             tab_q = double(tab_m(i,j))/G;
%                             dnum1 = round(yd/tab_q);
%                             if mod(dnum1,2)==0
%                                 e_code(n_msg)=0;
%                             else
%                                 e_code(n_msg)=1;
%                             end
%                         else
%                             break;
%                         end
%                         if e_code(n_msg)~=cover_lsb_change(n_msg)%����ȡ���������к�ԭ�������жԱȣ��������ͽ��ж�Ӧ������DCTϵ�����޸�
%                             dnum1 = round(yd/tab_q);
%                              if mod(dnum1,2)==0
%                                 dnum2 = floor(yd/tab_q);
%                                 if mod(dnum2,2)==1
%                                     yd = dnum2*tab_q;%��ȥ��ô��
%                                 else
%                                     yd = (dnum2+1)*tab_q;%������ô��
%                                 end                        
%                              else
%                                 dnum2 = floor(yd/tab_q);
%                                 if mod(dnum2,2)==1
%                                     yd = (dnum2+1)*tab_q;%������ô��
%                                 else
%                                     yd = dnum2*tab_q;%������ô��
%                                 end
%                              end  
%                             cover_DCT((bm-1)*8+i,(bn-1)*8+j)=yd;
%                         end
%                     end
%                 end
%             end
%             if n_msg>code_n break; end
%         end
%         if n_msg>code_n break; end
%     end  
% end
%% ��������ͼ��
cover_spa = blkproc(cover_DCT,[8 8],'P1*x*P2',t',t);
cover_spa = cover_spa + double(2^(bits-1));
cover_spa = uint8(cover_spa);
imwrite(cover_spa,stegoPath,'quality',cover_QF);

end