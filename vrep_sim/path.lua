function sysCall_init() 
    BillHandle=sim.getObjectHandle('Bill')

    legJointHandles={sim.getObjectHandle('Bill_leftLegJoint'),sim.getObjectHandle('Bill_rightLegJoint')}

    kneeJointHandles={sim.getObjectHandle('Bill_leftKneeJoint'),sim.getObjectHandle('Bill_rightKneeJoint')}

    ankleJointHandles={sim.getObjectHandle('Bill_leftAnkleJoint'),sim.getObjectHandle('Bill_rightAnkleJoint')}

    shoulderJointHandles={sim.getObjectHandle('Bill_leftShoulderJoint'),sim.getObjectHandle('Bill_rightShoulderJoint')}

    elbowJointHandles={sim.getObjectHandle('Bill_leftElbowJoint'),sim.getObjectHandle('Bill_rightElbowJoint')}

    wristJointHandles={sim.getObjectHandle('Bill_leftWristJoint'),sim.getObjectHandle('Bill_rightWristJoint')}

    neckJointHandle=sim.getObjectHandle('Bill_neck')
    headTopJointHandle=sim.getObjectHandle('Bill_headTop')

    pathHandle=sim.getObjectHandle('Bill_path')

    targetHandle=sim.getObjectHandle('Bill_goalDummy')

    pathPlanningHandle=simGetPathPlanningHandle('Bill_task')

    

    sim.setObjectParent(pathHandle,-1,true)

    sim.setObjectParent(targetHandle,-1,true)

    

    legWaypoints={0.237,0.228,0.175,-0.014,-0.133,-0.248,-0.323,-0.450,-0.450,-0.442,-0.407,-0.410,-0.377,-0.303,-0.178,-0.111,-0.010,0.046,0.104,0.145,0.188}

    kneeWaypoints={0.282,0.403,0.577,0.929,1.026,1.047,0.939,0.664,0.440,0.243,0.230,0.320,0.366,0.332,0.269,0.222,0.133,0.089,0.065,0.073,0.092}

    ankleWaypoints={-0.133,0.041,0.244,0.382,0.304,0.232,0.266,0.061,-0.090,-0.145,-0.043,0.041,0.001,0.011,-0.099,-0.127,-0.121,-0.120,-0.107,-0.100,-0.090,-0.009}

    shoulderWaypoints={0.028,0.043,0.064,0.078,0.091,0.102,0.170,0.245,0.317,0.337,0.402,0.375,0.331,0.262,0.188,0.102,0.094,0.086,0.080,0.051,0.058,0.048}

    elbowWaypoints={-1.148,-1.080,-1.047,-0.654,-0.517,-0.366,-0.242,-0.117,-0.078,-0.058,-0.031,-0.001,-0.009,0.008,-0.108,-0.131,-0.256,-0.547,-0.709,-0.813,-1.014,-1.102}

    relativeVel={2,2,1.2,2.3,1.4,1,1,1,1,1.6,1.9,2.4,2.0,1.9,1.5,1,1,1,1,1,2.3,1.5}

    

    nominalVelocity=sim.getScriptSimulationParameter(sim.handle_self,'walkingSpeed')

    randomColors=sim.getScriptSimulationParameter(sim.handle_self,'randomColors')

    scaling=0

    tl=#legWaypoints

    dl=1/tl

    vp=0

    desiredTargetPos={-99,-99}

    pathCalculated=0 -- 0=not calculated, 1=beeing calculated, 2=calculated

    tempPathSearchObject=-1

    currentPosOnPath=0

    

    HairColors={4,{0.30,0.22,0.14},{0.75,0.75,0.75},{0.075,0.075,0.075},{0.75,0.68,0.23}}

    skinColors={2,{0.61,0.54,0.45},{0.52,0.45,0.35}}

    shirtColors={5,{0.27,0.36,0.54},{0.54,0.27,0.27},{0.31,0.51,0.33},{0.46,0.46,0.46},{0.18,0.18,0.18}}

    trouserColors={2,{0.4,0.34,0.2},{0.12,0.12,0.12}}

    shoeColors={2,{0.12,0.12,0.12},{0.25,0.12,0.045}}

    

    -- Initialize to random colors if desired:

    if (randomColors) then

        -- First we just retrieve all objects in the model:

        previousSelection=sim.getObjectSelection()

        sim.removeObjectFromSelection(sim.handle_all,-1)

        sim.addObjectToSelection(sim.handle_tree,BillHandle)

        modelObjects=sim.getObjectSelection()

        sim.removeObjectFromSelection(sim.handle_all,-1)

        sim.addObjectToSelection(previousSelection)

        -- Now we set random colors:

        math.randomseed(sim.getFloatParameter(sim.floatparam_rand)*10000) -- each lua instance should start with a different and 'good' seed

        setColor(modelObjects,'HAIR',HairColors[1+math.random(HairColors[1])])

        setColor(modelObjects,'SKIN',skinColors[1+math.random(skinColors[1])])

        setColor(modelObjects,'SHIRT',shirtColors[1+math.random(shirtColors[1])])

        setColor(modelObjects,'TROUSERS',trouserColors[1+math.random(trouserColors[1])])

        setColor(modelObjects,'SHOE',shoeColors[1+math.random(shoeColors[1])])

    end
    
    pub=simROS.advertise('/joints', 'sensor_msgs/PointCloud')

end
------------------------------------------------------------------------------ 
-- Following few lines automatically added by V-REP to guarantee compatibility 
-- with V-REP 3.1.3 and earlier: 
colorCorrectionFunction=function(_aShapeHandle_) 
  local version=sim.getInt32Parameter(sim.intparam_program_version) 
  local revision=sim.getInt32Parameter(sim.intparam_program_revision) 
  if (version<30104)and(revision<3) then 
      return _aShapeHandle_ 
  end 
  return '@backCompatibility1:'.._aShapeHandle_ 
end 
------------------------------------------------------------------------------ 
 
 
setColor=function(objectTable,colorName,color)

    for i=1,#objectTable,1 do

        if (sim.getObjectType(objectTable[i])==sim.object_shape_type) then

            sim.setShapeColor(colorCorrectionFunction(objectTable[i]),colorName,0,color)

        end

    end

end





function sysCall_cleanup() 
    sim.setObjectParent(pathHandle,BillHandle,true)

    sim.setObjectParent(targetHandle,BillHandle,true)

    -- Restore to initial colors:

    if (randomColors) then

        previousSelection=sim.getObjectSelection()

        sim.removeObjectFromSelection(sim.handle_all,-1)

        sim.addObjectToSelection(sim.handle_tree,BillHandle)

        modelObjects=sim.getObjectSelection()

        sim.removeObjectFromSelection(sim.handle_all,-1)

        sim.addObjectToSelection(previousSelection)

        setColor(modelObjects,'HAIR',HairColors[2])

        setColor(modelObjects,'SKIN',skinColors[2])

        setColor(modelObjects,'SHIRT',shirtColors[2])

        setColor(modelObjects,'TROUSERS',trouserColors[2])

        setColor(modelObjects,'SHOE',shoeColors[2])

    end
    
    simROS.shutdownPublisher(pub)

end 


function sysCall_actuation() 
    s=sim.getObjectSizeFactor(BillHandle)

    

    -- Check if we need to recompute the path (e.g. because the goal position has moved):

    targetP=sim.getObjectPosition(targetHandle,-1)

    vv={targetP[1]-desiredTargetPos[1],targetP[2]-desiredTargetPos[2]}

    if (math.sqrt(vv[1]*vv[1]+vv[2]*vv[2])>0.01) then

        pathCalculated=0 -- We have to recompute the path since the target position has moved

        desiredTargetPos[1]=targetP[1]

        desiredTargetPos[2]=targetP[2]

    end

    

    rightV=0

    leftV=0

    

        if (pathCalculated==0) then

            -- We need to initialize a path search object:

            if (tempPathSearchObject~=-1) then

                -- delete any previous temporary path search object:    

                simPerformPathSearchStep(tempPathSearchObject,true) 

            end

            tempPathSearchObject=simInitializePathSearch(pathPlanningHandle,5,0.03) -- search for a maximum of 5 seconds

            if (tempPathSearchObject~=-1) then

                pathCalculated=1 -- Initialization went fine

            end

        else

            if (pathCalculated==1) then

                -- A path hasn't been found yet, we need to perform another path search step:

                r=simPerformPathSearchStep(tempPathSearchObject,false)

                if (r<1) then

                    -- Path was not yet found, or the search has failed

                    if (r~=-2) then

                        -- path search has failed!

                        pathCalculated=0

                        tempPathSearchObject=-1

                    end

                else

                    -- we found a path!

                    pathCalculated=2 

                    currentPosOnPath=0

                    tempPathSearchObject=-1

                end

            else

                -- We have an existing path. We follow that path:

                l=sim.getPathLength(pathHandle)

                r=sim.getObjectPosition(BillHandle,-1)

                while true do

                    p=sim.getPositionOnPath(pathHandle,currentPosOnPath/l)

                    d=math.sqrt((p[1]-r[1])*(p[1]-r[1])+(p[2]-r[2])*(p[2]-r[2]))

                    if (d>0.3)or(currentPosOnPath>=l) then

                        break

                    end

                    currentPosOnPath=currentPosOnPath+0.05

                end

                if (d>0.1) then

                    -- Ok, we follow the path

                    m=sim.getObjectMatrix(BillHandle,-1)

                    m=simGetInvertedMatrix(m)

                    p=sim.multiplyVector(m,p)

                    -- Now p is relative to the mannequin

                    a=math.atan2(p[2],p[1])

                    if (a>=0)and(a<math.pi*0.5) then

                        rightV=nominalVelocity

                        leftV=nominalVelocity*(1-2*a/(math.pi*0.5))

                    end

                    if (a>=math.pi*0.5) then

                        leftV=-nominalVelocity

                        rightV=nominalVelocity*(1-2*(a-math.pi*0.5)/(math.pi*0.5))

                    end

                    if (a<0)and(a>-math.pi*0.5) then

                        leftV=nominalVelocity

                        rightV=nominalVelocity*(1+2*a/(math.pi*0.5))

                    end

                    if (a<=-math.pi*0.5) then

                        rightV=-nominalVelocity

                        leftV=nominalVelocity*(1+2*(a+math.pi*0.5)/(math.pi*0.5))

                    end

                else

                    -- We arrived at the end of the path. The position of Bill still might not

                    -- coincide with the goal position if we selected the "partial path" option in the path planning dialog

                    targetP=sim.getObjectPosition(targetHandle,-1)

                    billP=sim.getObjectPosition(BillHandle,-1)

                    vv={targetP[1]-billP[1],targetP[2]-billP[2]}

                    if (math.sqrt(vv[1]*vv[1]+vv[2]*vv[2])>0.2) then

                        pathCalculated=0 -- We have to recompute the path

                    end

                end

            end

        end

    

    

    

    vel=(rightV+leftV)*0.5*0.8/0.56

    if (vel<0) then vel=0 end

    

    scaling=(vel/nominalVelocity)/1.4

    

    vp=vp+sim.getSimulationTimeStep()*vel

    p=math.fmod(vp,1)

    indexLow=math.floor(p/dl)

    t=p/dl-indexLow

    oppIndexLow=math.floor(indexLow+tl/2)

    if (oppIndexLow>=tl) then oppIndexLow=oppIndexLow-tl end

    indexHigh=indexLow+1

    if (indexHigh>=tl) then indexHigh=indexHigh-tl end

    oppIndexHigh=oppIndexLow+1

    if (oppIndexHigh>=tl) then oppIndexHigh=oppIndexHigh-tl end

    

    sim.setJointPosition(legJointHandles[1],(legWaypoints[indexLow+1]*(1-t)+legWaypoints[indexHigh+1]*t)*scaling)

    sim.setJointPosition(kneeJointHandles[1],(kneeWaypoints[indexLow+1]*(1-t)+kneeWaypoints[indexHigh+1]*t)*scaling)

    sim.setJointPosition(ankleJointHandles[1],(ankleWaypoints[indexLow+1]*(1-t)+ankleWaypoints[indexHigh+1]*t)*scaling)

    sim.setJointPosition(shoulderJointHandles[1],(shoulderWaypoints[indexLow+1]*(1-t)+shoulderWaypoints[indexHigh+1]*t)*scaling)

    sim.setJointPosition(elbowJointHandles[1],(elbowWaypoints[indexLow+1]*(1-t)+elbowWaypoints[indexHigh+1]*t)*scaling)

    

    sim.setJointPosition(legJointHandles[2],(legWaypoints[oppIndexLow+1]*(1-t)+legWaypoints[oppIndexHigh+1]*t)*scaling)

    sim.setJointPosition(kneeJointHandles[2],(kneeWaypoints[oppIndexLow+1]*(1-t)+kneeWaypoints[oppIndexHigh+1]*t)*scaling)

    sim.setJointPosition(ankleJointHandles[2],(ankleWaypoints[oppIndexLow+1]*(1-t)+ankleWaypoints[oppIndexHigh+1]*t)*scaling)

    sim.setJointPosition(shoulderJointHandles[2],(shoulderWaypoints[oppIndexLow+1]*(1-t)+shoulderWaypoints[oppIndexHigh+1]*t)*scaling)

    sim.setJointPosition(elbowJointHandles[2],(elbowWaypoints[oppIndexLow+1]*(1-t)+elbowWaypoints[oppIndexHigh+1]*t)*scaling)

    

    linMov=s*sim.getSimulationTimeStep()*(rightV+leftV)*0.5*scaling*(relativeVel[indexLow+1]*(1-t)+relativeVel[indexHigh+1]*t)

    rotMov=sim.getSimulationTimeStep()*math.atan((rightV-leftV)*8)

    position=sim.getObjectPosition(BillHandle,sim.handle_parent)

    orientation=sim.getObjectOrientation(BillHandle,sim.handle_parent)

    xDir={math.cos(orientation[3]),math.sin(orientation[3]),0.0}

    position[1]=position[1]+xDir[1]*linMov

    position[2]=position[2]+xDir[2]*linMov

    orientation[3]=orientation[3]+rotMov

    sim.setObjectPosition(BillHandle,sim.handle_parent,position)

    sim.setObjectOrientation(BillHandle,sim.handle_parent,orientation)
    
    msg={}
    msg['header']={seq=0,stamp=simROS.getTime(),frame_id='map'}
    msg['points']={}
    p1={}
    p1['x']=sim.getObjectPosition(headTopJointHandle, -1)[1]
    p1['y']=sim.getObjectPosition(headTopJointHandle, -1)[2]
    p1['z']=sim.getObjectPosition(headTopJointHandle, -1)[3]
    msg['points'][1]=p1
    p2={}
    p2['x']=sim.getObjectPosition(neckJointHandle, -1)[1]
    p2['y']=sim.getObjectPosition(neckJointHandle, -1)[2]
    p2['z']=sim.getObjectPosition(neckJointHandle, -1)[3]
    msg['points'][2]=p2
    p3={}
    p3['x']=sim.getObjectPosition(shoulderJointHandles[1], -1)[1]
    p3['y']=sim.getObjectPosition(shoulderJointHandles[1], -1)[2]
    p3['z']=sim.getObjectPosition(shoulderJointHandles[1], -1)[3]
    msg['points'][3]=p3
    p4={}
    p4['x']=sim.getObjectPosition(shoulderJointHandles[2], -1)[1]
    p4['y']=sim.getObjectPosition(shoulderJointHandles[2], -1)[2]
    p4['z']=sim.getObjectPosition(shoulderJointHandles[2], -1)[3]
    msg['points'][4]=p4
    p5={}
    p5['x']=sim.getObjectPosition(elbowJointHandles[1], -1)[1]
    p5['y']=sim.getObjectPosition(elbowJointHandles[1], -1)[2]
    p5['z']=sim.getObjectPosition(elbowJointHandles[1], -1)[3]
    msg['points'][5]=p5
    p6={}
    p6['x']=sim.getObjectPosition(elbowJointHandles[2], -1)[1]
    p6['y']=sim.getObjectPosition(elbowJointHandles[2], -1)[2]
    p6['z']=sim.getObjectPosition(elbowJointHandles[2], -1)[3]
    msg['points'][6]=p6
    p7={}
    p7['x']=sim.getObjectPosition(wristJointHandles[1], -1)[1]
    p7['y']=sim.getObjectPosition(wristJointHandles[1], -1)[2]
    p7['z']=sim.getObjectPosition(wristJointHandles[1], -1)[3]
    msg['points'][7]=p7
    p8={}
    p8['x']=sim.getObjectPosition(wristJointHandles[2], -1)[1]
    p8['y']=sim.getObjectPosition(wristJointHandles[2], -1)[2]
    p8['z']=sim.getObjectPosition(wristJointHandles[2], -1)[3]
    msg['points'][8]=p8
    p9={}
    p9['x']=sim.getObjectPosition(legJointHandles[1], -1)[1]
    p9['y']=sim.getObjectPosition(legJointHandles[1], -1)[2]
    p9['z']=sim.getObjectPosition(legJointHandles[1], -1)[3]
    msg['points'][9]=p9
    p10={}
    p10['x']=sim.getObjectPosition(legJointHandles[2], -1)[1]
    p10['y']=sim.getObjectPosition(legJointHandles[2], -1)[2]
    p10['z']=sim.getObjectPosition(legJointHandles[2], -1)[3]
    msg['points'][10]=p10
    p11={}
    p11['x']=sim.getObjectPosition(kneeJointHandles[1], -1)[1]
    p11['y']=sim.getObjectPosition(kneeJointHandles[1], -1)[2]
    p11['z']=sim.getObjectPosition(kneeJointHandles[1], -1)[3]
    msg['points'][11]=p11
    p12={}
    p12['x']=sim.getObjectPosition(kneeJointHandles[2], -1)[1]
    p12['y']=sim.getObjectPosition(kneeJointHandles[2], -1)[2]
    p12['z']=sim.getObjectPosition(kneeJointHandles[2], -1)[3]
    msg['points'][12]=p12
    p13={}
    p13['x']=sim.getObjectPosition(ankleJointHandles[1], -1)[1]
    p13['y']=sim.getObjectPosition(ankleJointHandles[1], -1)[2]
    p13['z']=sim.getObjectPosition(ankleJointHandles[1], -1)[3]
    msg['points'][13]=p13
    p14={}
    p14['x']=sim.getObjectPosition(ankleJointHandles[2], -1)[1]
    p14['y']=sim.getObjectPosition(ankleJointHandles[2], -1)[2]
    p14['z']=sim.getObjectPosition(ankleJointHandles[2], -1)[3]
    msg['points'][14]=p14
    
    simROS.publish(pub,msg)

end 

