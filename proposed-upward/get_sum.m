function [min_value, max_value, b1, b2] = get_sum(usable_DCT_num)
switch usable_DCT_num
    case 'E34' %7
        min_value = 4;
        max_value = 5;
        b1 = 2;
        b2 = 1;
    case 'E345' %12
        min_value = 4;
        max_value = 6;
        b1 = 4;
        b2 = 1;
    case 'E45' % 9
        min_value = 5;
        max_value = 6;
        b1 = 3;
        b2 = 1;
    case 'E456' % 15
        min_value = 5;
        max_value = 7;
        b1 = 6;
        b2 = 1;
    case 'E4567' % 22
        min_value = 5;
        max_value = 8;
        b1 = 9;
        b2 = 1;
    case 'E45678' %30
        min_value = 5;
        max_value = 9;
        b1 = 12;
        b2 = 2;
    case 'E56' %11
        min_value = 6;
        max_value = 7;
        b1 = 4;
        b2 = 1;
    case 'E567' % 18
        min_value = 6;
        max_value = 8;
        b1 = 7;
        b2 = 1;
    case 'E5678' % 26
        min_value = 6;
        max_value = 9;
        b1 = 10;
        b2 = 2;
    case 'E67' % 13
        min_value = 7;
        max_value = 8;
        b1 = 5;
        b2 = 1;
    case 'E678' % 21
        min_value = 7;
        max_value = 9;
        b1 = 8;
        b2 = 2;
%     case 24
%         min_vale = 8;
%         max_value = 10;
    case 'E78' %15
        min_value = 8;
        max_value = 9;
        b1 = 6;
        b2 = 1;
%     case 31
%         min_value = 5;
%         max_value = 9;
%         b1 = 11;
%         b2 = 3;
end
end