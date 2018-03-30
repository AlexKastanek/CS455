clc, clear
close all

%parameters
n = 100; %number of nodes
m = 2; %number of dimensions
k = 1.2; %ratio of optimal distance
d = 15; %optimal distance
r = k * d; %neighbor radius
delta_t = .008; 

%setting number of frames for video and initialzing video
nFrames = 1000;
mov(1:nFrames) = struct('cdata', [], 'colormap', []);
vid = VideoWriter('static-flocking.avi');
vid.FrameRate = 30;


%generating initial status for each sensor node
for i = 1:n
    sensorNode(i).xPos = randi(50);
    sensorNode(i).yPos = randi(50);
    sensorNode(i).xVel = 0;
    sensorNode(i).yVel = 0;
    sensorNode(i).xAcc = 0;
    sensorNode(i).yAcc = 0;
    sensorNode(i).q = [sensorNode(i).xPos sensorNode(i).yPos];
    sensorNode(i).qOld = [sensorNode(i).xPos sensorNode(i).yPos];
    sensorNode(i).p = [sensorNode(i).xVel sensorNode(i).yVel];
    sensorNode(i).u = [sensorNode(i).xAcc sensorNode(i).yAcc];
    sensorNode(i).trajectory = zeros(nFrames, 2);
    sensorNode(i).velocities = zeros(nFrames, 2);
end

%initializing position of gamma agent (sensor node target)
gammaAgent.q = [150 150];

for iteration = 1:nFrames
    
    %constructing list of neighbors and spacial adjacency matrix as well as
    %storing some additional data
    for i = 1:n
        %storing current trajectory and velocity of each node
        sensorNode(i).trajectory(iteration, 1) = sensorNode(i).q(1);
        sensorNode(i).trajectory(iteration, 2) = sensorNode(i).q(2);
        sensorNode(i).velocities(iteration, 1) = sensorNode(i).p(1);
        sensorNode(i).velocities(iteration, 2) = sensorNode(i).p(2);
        %initializing list of neighbors
        sensorNode(i).Neighbors = [];
        %counter used to increment number of neighbors
        counter = 1;
        for j = 1:n
            q_i = sensorNode(i).q;
            q_j = sensorNode(j).q;
            if i ~= j
                distance = norm(q_j - q_i);
                if distance < r
                	%j is a neighbor of i
                    sensorNode(i).Neighbors(counter) = j;
                    counter = counter + 1;
                end
                %setting adjacency matrix
                AdjMat(i,j) = bumpFunction(sigmaNorm(q_j - q_i)/sigmaNorm(r), 0.2);
            else
                AdjMat(i,j) = 0;
            end
        end
        sensorNode(i).neighborsSize = size(sensorNode(i).Neighbors, 2);
    end
    
    %plotting nodes
    plotNodes(sensorNode, n);
    
    %plotting borders
    hold on
    plot(gammaAgent.q(1), gammaAgent.q(2), 'go');
    plot(-50, -50, 'k.');
    plot(200, 200, 'k.');
    hold off;
    
    %creating a table to display the data of each node
    T = struct2table(sensorNode);
    
    %drawing a line between each neighboring sensor node
    plotLinesToNeighbors(sensorNode, n);
    
    %getting connectivity
    connectivity(iteration) = rank(AdjMat)/100;
    
    %the flocking algorithm
    for i = 1:n
        sensorNode(i).u = zeros(1,2); %acceleration of node i
        GBT = zeros(sensorNode(i).neighborsSize, 2); %Gradient-Based Term
        CT = zeros(sensorNode(i).neighborsSize, 2); %Consensus Term
        for j = 1:sensorNode(i).neighborsSize
            q_i = sensorNode(i).q;
            p_i = sensorNode(i).p;
            q_j = sensorNode(sensorNode(i).Neighbors(j)).q;
            p_j = sensorNode(sensorNode(i).Neighbors(j)).p;
            GBT(j, :) = phiAlphaPotential(sigmaNorm(q_j - q_i),d,r)
            			*sigmaGradient(q_j - q_i);
            CT(j, :) = AdjMat(i,sensorNode(i).Neighbors(j))*(p_j - p_i);
        end
        sensorNode(i).u = 30*sum(GBT) + 12*sum(CT) 
        				  - 1.1*(sensorNode(i).q - gammaAgent.q) 
        				  - 2*sensorNode(i).p;
        %clearing the GBT and CT
        GBT = [];
        CT = [];
    end
    
    %setting frame iteration to the currently displayed figure
    hold on;
    mov(iteration) = getframe;
    hold off;
    
    clf;
    
    %setting new position and velocity of each node
    for i = 1:n
        sensorNode(i).qOld = sensorNode(i).q;
        sensorNode(i).q = sensorNode(i).qOld + sensorNode(i).p*delta_t 
        				  + (1/2)*sensorNode(i).u*(delta_t.^2);
        sensorNode(i).p = zeros(1,2);
        sensorNode(i).p = (sensorNode(i).q - sensorNode(i).qOld)/delta_t;
    end
    
end

%writing the frames to the video
open(vid);
writeVideo(vid, mov);
close(vid);

plotTrajectory(sensorNode, nFrames); %plotting trajectory of all nodes
%plotSingleTrajectory(sensorNode(50), nFrames); %plotting trajectory of node 50
plotVelocities(sensorNode, nFrames, n); %plotting velocities of all nodes
plotConnectivity(connectivity, nFrames); %plotting connectivity of node network

%%%%%%%%%%%%%% MSN FUNCTIONS %%%%%%%%%%%%%%

function f = phiAlphaPotential(z,d,r)
    f = bumpFunction(z/sigmaNorm(r),0.2)*phiPotential(z - sigmaNorm(d));
end

function f = phiPotential(z)
    a = 5;
    b = 5;
    c = (abs(a - b))/(sqrt(4*a*b));
    f = (1/2)*((a + b)*sigmaOne(z + c) + (a - b));
end

function f = sigmaOne(z)
    f = z/sqrt(1 + z^2);
end

function f = bumpFunction(z, h)
    if z >= 0 && z < h
        f = 1;
    elseif z >= h && z <= 1
        f = (1/2)*(1 + cos(pi*((z - h)/(1 - h))));
    else
        f = 0;
    end
end

function f = sigmaGradient(z)
    e = 0.1;
    f = z / sqrt(1 + e * (norm(z))^2);
end

function f = sigmaNorm(z)
    e = 0.1;
    f = (1/e)*(sqrt(1 + e*(norm(z))^2) - 1);
end

%%%%%%%%%%%%%% GRAPHING FUNCTIONS %%%%%%%%%%%%%%

function plotNodes(sensorNode,n)
    for i = 1:n
        hold on;
        plot(sensorNode(i).q(1), sensorNode(i).q(2), 'rx');
        hold off;
    end
end

function plotLinesToNeighbors(sensorNode,n)

    for i = 1:n
        for j = 1:sensorNode(i).neighborsSize
            x = [sensorNode(i).q(1) sensorNode(sensorNode(i).Neighbors(j)).q(1)];
            y = [sensorNode(i).q(2) sensorNode(sensorNode(i).Neighbors(j)).q(2)];
            hold on;
            line(x,y);
            hold off;
        end
    end

end

function plotTrajectory(sensorNode, nFrames, n)
    
    figure
    for i = 1:n
        for j = 1:nFrames
            hold on;
            if j == nFrames
                plot(sensorNode(i).trajectory(j,1), 
                	 sensorNode(i).trajectory(j,2), 'mx');
            else
                plot(sensorNode(i).trajectory(j,1), 
                	 sensorNode(i).trajectory(j,2), 'k.');
            end
            hold off;
            if j ~= 1
                x = [sensorNode(i).trajectory(j-1, 1) 
                	 sensorNode(i).trajectory(j, 1)];
                y = [sensorNode(i).trajectory(j-1, 2) 
                	 sensorNode(i).trajectory(j, 2)];
                hold on;
                line(x,y, 'Color', 'k');
                hold off;
            end
        end
    end
end

function plotSingleTrajectory(node, nFrames)

    figure
    for i = 1:nFrames
        hold on;
        plot(node.trajectory(i,1), node.trajectory(i,2), 'k.');
        hold off;
        if i ~= 1
            x = [node.trajectory(i-1,1) node.trajectory(i,1)];
            y = [node.trajectory(i-1,2) node.trajectory(i,2)];
            hold on;
            line(x,y,'Color', 'g', 'LineStyle', '--');
        end
    end
end

function plotVelocities(sensorNode, nFrames, n)

    figure
    for i = 1:n
        x = [];
        y = [];
        for j = 1:nFrames
            x(j) = j;
            y(j) = norm(sensorNode(i).velocities(j));
        end
        hold on;
        plot(x,y, 'DisplayName', sprintf('%d', i));
        hold off;
    end
end

function plotConnectivity(connectivity, nFrames)

    figure
    for i = 1:nFrames
        x(i) = i;
        y(i) = connectivity(i);
    end
    hold on
    plot(x,y);
    hold off;
end
