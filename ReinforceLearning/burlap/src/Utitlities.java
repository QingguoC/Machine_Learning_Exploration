import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.RenderedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Utitlities {
    public static void writeEvalToFile(String[][] logs, String[] header, String fileName){

        StringBuilder builder = new StringBuilder();
        for(int i = 0; i < header.length; i++)//for each column
        {
            builder.append(header[i]);//append to the output string
            if(i < header.length - 1)//if this is not the last row element
                builder.append(",");//then add comma
        }
        builder.append("\n");

        for(int i = 0; i < logs.length; i++)//for each iteration
        {
            for(int j = 0; j < logs[0].length; j++)//for each RO
            {
                builder.append(logs[i][j]);//append to the output string
                if(j < logs[0].length - 1)//if this is not the last row element
                    builder.append(",");//then add comma
            }
            builder.append("\n");//append new line at the end of the row
        }
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter("out/" + fileName + ".txt"));
            writer.write(builder.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    public static int calRewards(Episode e){
        int r = 0;
        for (int i = 0; i < e.rewardSequence.size(); i++){
            r += e.rewardSequence.get(i);
        }
        return r;
    }

    public static void manualValueFunctionVis(ValueFunction valueFunction, Policy p, int size, String name, State initialState, SADomain domain, SimpleHashableStateFactory hashingFactory){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);

        //define color function
        LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
        rb.addNextLandMark(0., Color.RED);
        rb.addNextLandMark(1., Color.BLUE);

        //define a 2D painter of state values, specifying
        //which variables correspond to the x and y coordinates of the canvas
        StateValuePainter2D svp = new StateValuePainter2D(rb);
        svp.setXYKeys("agent:x", "agent:y",
                new VariableDomain(0, size), new VariableDomain(0, size),
                1, 1);

        //create our ValueFunctionVisualizer that paints for all states
        //using the ValueFunction source and the state value painter we defined
        ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(allStates, svp, valueFunction);

        //define a policy painter that uses arrow glyphs for each of the grid world actions
        PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
        spp.setXYKeys("agent:x", "agent:y",
                new VariableDomain(0, size), new VariableDomain(0, size),
                1, 1);

        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
        spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
        spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


        //add our policy renderer to it
        gui.setSpp(spp);
        gui.setPolicy(p);
        //gui.setSize(1,1);
        //set the background color for places where states are not rendered to grey
        gui.setBgColor(Color.GRAY);
        gui.setTitle(name);
        //start it
        gui.initGUI();
    }
}

