x = [1:25];
y = [0.2941 0.3549 0.4077 0.4432 0.4927 0.5201 0.5478 0.5694 0.5875 0.6097 0.6295 0.6466 0.6663 0.6812 0.7031 0.7192 0.7332 0.7463 0.7657 0.7759 0.7868 0.7994 0.8126 0.8233 0.8315];
yv = [0.2214 0.2912 0.3561 0.4211 0.4497 0.4810 0.5189 0.5317 0.5551 0.5779 0.5931 0.5939 0.6024 0.6046 0.6133 0.6129 0.6148 0.6209 0.6205 0.6219 0.6247 0.6224 0.6258 0.6273 0.6265];
plot(x, y, '.-r', x, yv, '.-b');
xlabel('Number of Epoch');
ylabel('Accuracy');
axis([0 30 0 1]);
title('Training Procedure');
for i = 1 : 25
    text(x(i) - 0.5, y(i) + 0.05, sprintf('%0.2f', y(i)), 'FontSize', 4, 'color', 'red');
    text(x(i) - 0.5, yv(i) - 0.05, sprintf('%0.2f', yv(i)), 'FontSize', 4, 'color', 'blue');
end
legend('train','validation','location','southeast');
