M = csvread('./Results/10_30_YrVsTpScore.csv');

M = M(2:end, :);
a = sum(M, 2);
X = [1:10];
Y = [2008:2016];
mkdir pics
for i = 1:size(M,1)
    M(i,:) = M(i,:)/a(i);
    figure(i)
    plot(X, M(i,:));
    title(sprintf('Year %d', Y(i)));
    xlabel('Topic ID');
    ylabel('Score');
    grid on;
    saveas(gcf,['./pics/' sprintf('Year %d', Y(i)) '.png'])
end

for i = 1:size(M,2)    
    figure(i)
    plot(Y, M(:,i));
    title(sprintf('Topic ID %d', X(i)));
    grid on;
    saveas(gcf,['./pics/' sprintf('Topic ID %d', X(i)) '.png'])
end

figure();
% mesh(X,Y,M);
b = sum(M);
plot(b)

grid on
xlabel('Topic ID')
ylabel('Score')
title('Comprehensive Analysis: 10 Topics, 30 Pass')