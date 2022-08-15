function recompression(coverPath,cover_QF,recompress_Path,tab_m,attack_QF,cover_lsb_change)%�����޸Ĳ���

%%  DCT�任

bits = 8;     
cover_spa = imread(coverPath);
cover_spa = double(cover_spa) - 2^(round(bits)-1);
[xm,xn] = size(cover_spa);
t = dctmtx(8);  %����DCT����
fun = @(xl) (t*xl*(t'));
cover_DCT = blkproc(cover_spa,[8 8],fun);%�ֿ�DCT�任
m_block = floor(xm/8);
n_block = floor(xn/8);


G=1;

%% �޸Ľ׶� �޸Ĳ��ȶ���DCTϵ��
for i=1:2
    cover_spa = blkproc(cover_DCT,[8 8],'P1*x*P2',t',t);
    cover_spa = cover_spa + double(2^(bits-1));
    cover_spa = uint8(cover_spa);
    imwrite(cover_spa,recompress_Path,'quality',attack_QF);
    bits = 8;     
    cover_spa = imread(recompress_Path);
    cover_spa = double(cover_spa) - 2^(round(bits)-1);
    t = dctmtx(8);  %����DCT����
    fun = @(xl) (t*xl*(t'));
    cover_DCT = blkproc(cover_spa,[8 8],fun);%�ֿ�DCT�任
    n_msg = 0;
    code_n = m_block*n_block*9;  %�е�Ƶ9��DCTϵ��
    e_code = zeros(1,code_n);
    for bm = 1:m_block
        for bn = 1:n_block
            for i = 1:8
                for j = 1:8
                    if (i+j==5)||(i+j==6)  %��Ƶ9��DCTϵ�� ��ȡ��������
                        n_msg = n_msg + 1;
                         if n_msg<=code_n
                            yd = cover_DCT((bm-1)*8+i,(bn-1)*8+j);
                            tab_q = double(tab_m(i,j))/G;
                            dnum1 = round(yd/tab_q);
                            if mod(dnum1,2)==0
                                e_code(n_msg)=0;
                            else
                                e_code(n_msg)=1;
                            end
                        else
                            break;
                        end
                        if e_code(n_msg)~=cover_lsb_change(n_msg)%����ȡ���������к�ԭ�������жԱȣ��������ͽ��ж�Ӧ������DCTϵ�����޸�
                            dnum1 = round(yd/tab_q);
                             if mod(dnum1,2)==0
                                dnum2 = floor(yd/tab_q);
                                if mod(dnum2,2)==1
                                    yd = dnum2*tab_q;%��ȥ��ô��
                                else
                                    yd = (dnum2+1)*tab_q;%������ô��
                                end                        
                             else
                                dnum2 = floor(yd/tab_q);
                                if mod(dnum2,2)==1
                                    yd = (dnum2+1)*tab_q;%������ô��
                                else
                                    yd = dnum2*tab_q;%������ô��
                                end
                             end  
                            cover_DCT((bm-1)*8+i,(bn-1)*8+j)=yd;
                        end
                    end
                end
            end
            if n_msg>code_n break; end
        end
        if n_msg>code_n break; end
    end  
end
%% ��������ͼ��
cover_spa = blkproc(cover_DCT,[8 8],'P1*x*P2',t',t);
cover_spa = cover_spa + double(2^(bits-1));
cover_spa = uint8(cover_spa);
imwrite(cover_spa,coverPath,'quality',cover_QF);

end