/*
 * comsol_example.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

public class comsol_example {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    /** model parameter **/
    model.param().set("L_tun", "350[mm]");
    model.param().set("L_cav", "120[mm]");
    model.param().set("H_cav_out", "95[mm]");
    model.param().set("W_cav_out", "138[mm]");
    model.param().set("H_cav_in", "43[mm]");
    model.param().set("W_cav_in", "86[mm]");
    model.param().set("R_cyl", "30[mm]");
    model.param().set("R_tube", "5[mm]");
    model.param().set("R_tube_od", "6[mm]");
    model.param().set("R_major", "20[mm]");
    model.param().set("f_sup", "2.45[GHz]");
    model.param().set("L_slide", "200[mm]");
    model.param().set("L_stub1", "25.0[mm]");
    model.param().set("L_stub2", "25.0[mm]");
    model.param().set("L_stub3", "25.0[mm]");
    model.param().set("P_supply", "100[W]");
    model.param().set("T0", "293.15[K]");

    /** Create plain component, geometry, and mesh **/
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3); /** 3 denotes three-dimension **/
    model.component("comp1").mesh().create("mesh1");

    /** Create geometry **/
    model.component("comp1").geom("geom1").create("wp1", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp1").set("quickplane", "yz");
    model.component("comp1").geom("geom1").feature("wp1").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp1").geom().create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("r1").set("base", "center");
    model.component("comp1").geom("geom1").feature("wp1").geom().feature("r1")
         .set("size", new String[]{"W_cav_in", "H_cav_in"});
    model.component("comp1").geom("geom1").create("ext1", "Extrude");
    model.component("comp1").geom("geom1").feature("ext1").label("cavity_mid");
    model.component("comp1").geom("geom1").feature("ext1").setIndex("distance", "L_cav", 0);
    model.component("comp1").geom("geom1").feature("ext1").selection("input").set("wp1");
    model.component("comp1").geom("geom1").create("wp2", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp2").set("quickplane", "yz");
    model.component("comp1").geom("geom1").feature("wp2").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp2").geom().create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("base", "center");
    model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
         .set("size", new String[]{"W_cav_in", "H_cav_in"});
    model.component("comp1").geom("geom1").create("ext2", "Extrude");
    model.component("comp1").geom("geom1").feature("ext2").label("Stub_tuner");
    model.component("comp1").geom("geom1").feature("ext2").setIndex("distance", "-L_tun", 0);
    model.component("comp1").geom("geom1").feature("ext2").selection("input").set("wp2");
    model.component("comp1").geom("geom1").create("wp4", "WorkPlane");
    model.component("comp1").geom("geom1").feature("wp4").set("quickplane", "yz");
    model.component("comp1").geom("geom1").feature("wp4").set("quickx", "L_cav");
    model.component("comp1").geom("geom1").feature("wp4").set("unite", true);
    model.component("comp1").geom("geom1").feature("wp4").geom().create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("wp4").geom().feature("r1").set("base", "center");
    model.component("comp1").geom("geom1").feature("wp4").geom().feature("r1")
         .set("size", new String[]{"W_cav_in", "H_cav_in"});
    model.component("comp1").geom("geom1").create("ext4", "Extrude");
    model.component("comp1").geom("geom1").feature("ext4").label("Sliding Circuit");
    model.component("comp1").geom("geom1").feature("ext4").setIndex("distance", "L_slide", 0);
    model.component("comp1").geom("geom1").feature("ext4").selection("input").set("wp4");
    model.component("comp1").geom("geom1").create("cyl1", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl1").label("cavity_top");
    model.component("comp1").geom("geom1").feature("cyl1").set("pos", new String[]{"L_cav/2", "0", "H_cav_in/2"});
    model.component("comp1").geom("geom1").feature("cyl1").set("r", "R_cyl");
    model.component("comp1").geom("geom1").feature("cyl1").set("h", "(H_cav_out-H_cav_in)/2");
    model.component("comp1").geom("geom1").create("cyl2", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl2").label("cavity_bot");
    model.component("comp1").geom("geom1").feature("cyl2").set("pos", new String[]{"L_cav/2", "0", "-H_cav_in/2"});
    model.component("comp1").geom("geom1").feature("cyl2").set("axis", new int[]{0, 0, -1});
    model.component("comp1").geom("geom1").feature("cyl2").set("r", "R_cyl");
    model.component("comp1").geom("geom1").feature("cyl2").set("h", "(H_cav_out-H_cav_in)/2");
    model.component("comp1").geom("geom1").create("uni1", "Union");
    model.component("comp1").geom("geom1").feature("uni1").selection("input")
         .set("cyl1", "cyl2", "ext1", "ext2", "ext4");
    model.component("comp1").geom("geom1").create("uni2", "Union");
    model.component("comp1").geom("geom1").feature("uni2").label("cavity");
    model.component("comp1").geom("geom1").feature("uni2").selection("input").set("uni1");
    model.component("comp1").geom("geom1").create("cyl4", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl4").label("stub1");
    model.component("comp1").geom("geom1").feature("cyl4").set("pos", new String[]{"-60[mm]", "0", "H_cav_in/2"});
    model.component("comp1").geom("geom1").feature("cyl4").set("axis", new int[]{0, 0, -1});
    model.component("comp1").geom("geom1").feature("cyl4").set("r", "20[mm]");
    model.component("comp1").geom("geom1").feature("cyl4").set("h", "L_stub1");
    model.component("comp1").geom("geom1").create("cyl5", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl5").label("stub2");
    model.component("comp1").geom("geom1").feature("cyl5").set("pos", new String[]{"-175[mm]", "0", "H_cav_in/2"});
    model.component("comp1").geom("geom1").feature("cyl5").set("axis", new int[]{0, 0, -1});
    model.component("comp1").geom("geom1").feature("cyl5").set("r", "20[mm]");
    model.component("comp1").geom("geom1").feature("cyl5").set("h", "L_stub2");
    model.component("comp1").geom("geom1").create("cyl6", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl6").label("stub3");
    model.component("comp1").geom("geom1").feature("cyl6").set("pos", new String[]{"-290[mm]", "0", "H_cav_in/2"});
    model.component("comp1").geom("geom1").feature("cyl6").set("axis", new int[]{0, 0, -1});
    model.component("comp1").geom("geom1").feature("cyl6").set("r", "20[mm]");
    model.component("comp1").geom("geom1").feature("cyl6").set("h", "L_stub3");
    model.component("comp1").geom("geom1").create("cyl7", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl7").set("pos", new String[]{"L_cav/2", "0", "-H_cav_out/2"});
    model.component("comp1").geom("geom1").feature("cyl7").set("r", "R_tube");
    model.component("comp1").geom("geom1").feature("cyl7").set("h", "H_cav_out");
    model.component("comp1").geom("geom1").create("cyl8", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl8").set("pos", new String[]{"L_cav/2", "0", "-H_cav_out/2"});
    model.component("comp1").geom("geom1").feature("cyl8").set("r", "R_tube_od");
    model.component("comp1").geom("geom1").feature("cyl8").set("h", "H_cav_out");
    model.component("comp1").geom("geom1").create("dif1", "Difference");
    model.component("comp1").geom("geom1").feature("dif1").label("ReactorWall");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("cyl8");
    model.component("comp1").geom("geom1").feature("dif1").selection("input2").set("cyl7");
    model.component("comp1").geom("geom1").create("cyl9", "Cylinder");
    model.component("comp1").geom("geom1").feature("cyl9").label("Reactor");
    model.component("comp1").geom("geom1").feature("cyl9").set("pos", new String[]{"L_cav/2", "0", "-H_cav_out/2"});
    model.component("comp1").geom("geom1").feature("cyl9").set("r", "R_tube");
    model.component("comp1").geom("geom1").feature("cyl9").set("h", "H_cav_out");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").create("sel1", "ExplicitSelection");
    model.component("comp1").geom("geom1").feature("sel1").label("Air");
    model.component("comp1").geom("geom1").feature("sel1").selection("selection").set("fin(1)", 1, 5, 6, 7, 14);
    model.component("comp1").geom("geom1").create("sel2", "ExplicitSelection");
    model.component("comp1").geom("geom1").feature("sel2").label("Stub");
    model.component("comp1").geom("geom1").feature("sel2").selection("selection").set("fin(1)", 2, 3, 4);
    model.component("comp1").geom("geom1").create("sel3", "ExplicitSelection");
    model.component("comp1").geom("geom1").feature("sel3").label("Solvent");
    model.component("comp1").geom("geom1").feature("sel3").selection("selection").set("fin(1)", 11, 12, 13);
    model.component("comp1").geom("geom1").create("sel4", "ExplicitSelection");
    model.component("comp1").geom("geom1").feature("sel4").label("Tube");
    model.component("comp1").geom("geom1").feature("sel4").selection("selection").set("fin(1)", 8, 9, 10);
    model.component("comp1").geom("geom1").run();

    /**Create material with it's property**/
    model.component("comp1").material().create("mat2", "Common");
    model.component("comp1").material().create("mat3", "Common");
    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material().create("mat6", "Common");
    model.component("comp1").material().create("mat7", "Common");

    /** Select geometry **/
    model.component("comp1").material("mat2").selection().named("geom1_sel3");
    model.component("comp1").material("mat3").selection().named("geom1_sel1");
    model.component("comp1").material("mat1").selection().named("geom1_sel4");
    model.component("comp1").material("mat6").selection().set(2, 3, 4);
    model.component("comp1").material("mat7").selection().geom("geom1", 2);
    model.component("comp1").material("mat7").selection()
         .set(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78);

    /**Add property**/
    model.component("comp1").material("mat2").propertyGroup("def").func().create("eta", "Piecewise");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("Cp", "Piecewise");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("rho", "Piecewise");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("k", "Piecewise");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("cs", "Interpolation");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("int1", "Interpolation");
    model.component("comp1").material("mat2").propertyGroup("def").func().create("int2", "Interpolation");
    model.component("comp1").material("mat3").propertyGroup("def").func().create("eta", "Piecewise");
    model.component("comp1").material("mat3").propertyGroup("def").func().create("Cp", "Piecewise");
    model.component("comp1").material("mat3").propertyGroup("def").func().create("rho", "Analytic");
    model.component("comp1").material("mat3").propertyGroup("def").func().create("k", "Piecewise");
    model.component("comp1").material("mat3").propertyGroup("def").func().create("cs", "Analytic");
    model.component("comp1").material("mat3").propertyGroup().create("RefractiveIndex", "Refractive index");
    model.component("comp1").material("mat7").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material("mat7").propertyGroup().create("Murnaghan", "Murnaghan");
    model.component("comp1").material("mat7").propertyGroup().create("Lame", "Lam\u00e9 parameters");
    model.component("comp1").material("mat2").label("Water, liquid");
    model.component("comp1").material("mat2").set("family", "water");
    model.component("comp1").material("mat2").propertyGroup("def").func("eta").set("arg", "T");
    model.component("comp1").material("mat2").propertyGroup("def").func("eta")
         .set("pieces", new String[][]{{"273.15", "413.15", "1.3799566804-0.021224019151*T^1+1.3604562827E-4*T^2-4.6454090319E-7*T^3+8.9042735735E-10*T^4-9.0790692686E-13*T^5+3.8457331488E-16*T^6"}, {"413.15", "553.75", "0.00401235783-2.10746715E-5*T^1+3.85772275E-8*T^2-2.39730284E-11*T^3"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("Cp").set("arg", "T");
    model.component("comp1").material("mat2").propertyGroup("def").func("Cp")
         .set("pieces", new String[][]{{"273.15", "553.75", "12010.1471-80.4072879*T^1+0.309866854*T^2-5.38186884E-4*T^3+3.62536437E-7*T^4"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("rho").set("arg", "T");
    model.component("comp1").material("mat2").propertyGroup("def").func("rho")
         .set("pieces", new String[][]{{"273.15", "553.75", "838.466135+1.40050603*T^1-0.0030112376*T^2+3.71822313E-7*T^3"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("k").set("arg", "T");
    model.component("comp1").material("mat2").propertyGroup("def").func("k")
         .set("pieces", new String[][]{{"273.15", "553.75", "-0.869083936+0.00894880345*T^1-1.58366345E-5*T^2+7.97543259E-9*T^3"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("cs")
         .set("table", new String[][]{{"273", "1403"}, 
         {"278", "1427"}, 
         {"283", "1447"}, 
         {"293", "1481"}, 
         {"303", "1507"}, 
         {"313", "1526"}, 
         {"323", "1541"}, 
         {"333", "1552"}, 
         {"343", "1555"}, 
         {"353", "1555"}, 
         {"363", "1550"}, 
         {"373", "1543"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("cs").set("interp", "piecewisecubic");
    model.component("comp1").material("mat2").propertyGroup("def").func("int1").set("funcname", "eps_r");
    model.component("comp1").material("mat2").propertyGroup("def").func("int1")
         .set("table", new String[][]{{"293.15", "78.0"}, 
         {"303.15", "75.0"}, 
         {"313.15", "72.0"}, 
         {"323.15", "69.0"}, 
         {"333.15", "66.2"}, 
         {"343.15", "63.9"}, 
         {"353.15", "62.7"}, 
         {"363.15", "62.3"}, 
         {"373.15", "62.0"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("int1").set("extrap", "linear");
    model.component("comp1").material("mat2").propertyGroup("def").func("int1").set("argunit", "K");
    model.component("comp1").material("mat2").propertyGroup("def").func("int2").set("funcname", "eps_j");
    model.component("comp1").material("mat2").propertyGroup("def").func("int2")
         .set("table", new String[][]{{"293.15", "10.5"}, 
         {"303.15", "8.6"}, 
         {"313.15", "6.7"}, 
         {"323.15", "5.1"}, 
         {"333.15", "3.85"}, 
         {"343.15", "3.3"}, 
         {"353.15", "3.1"}, 
         {"363.15", "3.0"}, 
         {"373.15", "2.9"}});
    model.component("comp1").material("mat2").propertyGroup("def").func("int2").set("extrap", "linear");
    model.component("comp1").material("mat2").propertyGroup("def").func("int2").set("argunit", "K");
    model.component("comp1").material("mat2").propertyGroup("def").set("dynamicviscosity", "eta(T[1/K])[Pa*s]");
    model.component("comp1").material("mat2").propertyGroup("def").set("ratioofspecificheat", "1.0");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("electricconductivity", new String[]{"5.5e-6[S/m]", "0", "0", "0", "5.5e-6[S/m]", "0", "0", "0", "5.5e-6[S/m]"});
    model.component("comp1").material("mat2").propertyGroup("def").set("heatcapacity", "Cp(T[1/K])[J/(kg*K)]");
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "rho(T[1/K])[kg/m^3]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("thermalconductivity", new String[]{"k(T[1/K])", "0", "0", "0", "k(T[1/K])", "0", "0", "0", "k(T[1/K])"});
    model.component("comp1").material("mat2").propertyGroup("def").set("soundspeed", "cs(T[1/K])[m/s]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermittivity", new String[]{"eps_r(T)-eps_j(T)*j", "0", "0", "0", "eps_r(T)-eps_j(T)*j", "0", "0", "0", "eps_r(T)-eps_j(T)*j"});
    model.component("comp1").material("mat2").propertyGroup("def").addInput("temperature");
    model.component("comp1").material("mat3").label("Air");
    model.component("comp1").material("mat3").set("family", "air");
    model.component("comp1").material("mat3").propertyGroup("def").func("eta").set("arg", "T");
    model.component("comp1").material("mat3").propertyGroup("def").func("eta")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("Cp").set("arg", "T");
    model.component("comp1").material("mat3").propertyGroup("def").func("Cp")
         .set("pieces", new String[][]{{"200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("rho")
         .set("expr", "pA*0.02897/R_const[K*mol/J]/T");
    model.component("comp1").material("mat3").propertyGroup("def").func("rho").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat3").propertyGroup("def").func("rho").set("dermethod", "manual");
    model.component("comp1").material("mat3").propertyGroup("def").func("rho")
         .set("argders", new String[][]{{"pA", "d(pA*0.02897/R_const/T,pA)"}, {"T", "d(pA*0.02897/R_const/T,T)"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("rho")
         .set("plotargs", new String[][]{{"pA", "0", "1"}, {"T", "0", "1"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("k").set("arg", "T");
    model.component("comp1").material("mat3").propertyGroup("def").func("k")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("cs").set("expr", "sqrt(1.4*287*T)");
    model.component("comp1").material("mat3").propertyGroup("def").func("cs").set("args", new String[]{"T"});
    model.component("comp1").material("mat3").propertyGroup("def").func("cs").set("dermethod", "manual");
    model.component("comp1").material("mat3").propertyGroup("def").func("cs")
         .set("argders", new String[][]{{"T", "d(sqrt(1.4*287*T),T)"}});
    model.component("comp1").material("mat3").propertyGroup("def").func("cs")
         .set("plotargs", new String[][]{{"T", "0", "1"}});
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat3").propertyGroup("def").set("dynamicviscosity", "eta(T[1/K])[Pa*s]");
    model.component("comp1").material("mat3").propertyGroup("def").set("ratioofspecificheat", "1.4");
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("electricconductivity", new String[]{"0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"});
    model.component("comp1").material("mat3").propertyGroup("def").set("heatcapacity", "Cp(T[1/K])[J/(kg*K)]");
    model.component("comp1").material("mat3").propertyGroup("def").set("density", "rho(pA[1/Pa],T[1/K])[kg/m^3]");
    model.component("comp1").material("mat3").propertyGroup("def")
         .set("thermalconductivity", new String[]{"k(T[1/K])[W/(m*K)]", "0", "0", "0", "k(T[1/K])[W/(m*K)]", "0", "0", "0", "k(T[1/K])[W/(m*K)]"});
    model.component("comp1").material("mat3").propertyGroup("def").set("soundspeed", "cs(T[1/K])[m/s]");
    model.component("comp1").material("mat3").propertyGroup("def").addInput("temperature");
    model.component("comp1").material("mat3").propertyGroup("def").addInput("pressure");
    model.component("comp1").material("mat3").propertyGroup("RefractiveIndex").set("n", "");
    model.component("comp1").material("mat3").propertyGroup("RefractiveIndex").set("ki", "");
    model.component("comp1").material("mat3").propertyGroup("RefractiveIndex")
         .set("n", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat3").propertyGroup("RefractiveIndex")
         .set("ki", new String[]{"0", "0", "0", "0", "0", "0", "0", "0", "0"});
    model.component("comp1").material("mat1").label("PTFE");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermittivity", new String[]{"2.06", "0", "0", "0", "2.06", "0", "0", "0", "2.06"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("electricconductivity", new String[]{"0", "0", "0", "0", "0", "0", "0", "0", "0"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalconductivity", new String[]{"0.245", "0", "0", "0", "0.245", "0", "0", "0", "0.245"});
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "2160");
    model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", "1300");
    model.component("comp1").material("mat1").propertyGroup("def").set("ratioofspecificheat", "1");
    model.component("comp1").material("mat6").label("Copper");
    model.component("comp1").material("mat6").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat6").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat6").propertyGroup("def")
         .set("electricconductivity", new String[]{"5.998e7[S/m]", "0", "0", "0", "5.998e7[S/m]", "0", "0", "0", "5.998e7[S/m]"});
    model.component("comp1").material("mat7").label("Aluminum 1");
    model.component("comp1").material("mat7").set("family", "aluminum");
    model.component("comp1").material("mat7").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat7").propertyGroup("def").set("heatcapacity", "900[J/(kg*K)]");
    model.component("comp1").material("mat7").propertyGroup("def")
         .set("thermalconductivity", new String[]{"238[W/(m*K)]", "0", "0", "0", "238[W/(m*K)]", "0", "0", "0", "238[W/(m*K)]"});
    model.component("comp1").material("mat7").propertyGroup("def")
         .set("electricconductivity", new String[]{"3.774e7[S/m]", "0", "0", "0", "3.774e7[S/m]", "0", "0", "0", "3.774e7[S/m]"});
    model.component("comp1").material("mat7").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat7").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"23e-6[1/K]", "0", "0", "0", "23e-6[1/K]", "0", "0", "0", "23e-6[1/K]"});
    model.component("comp1").material("mat7").propertyGroup("def").set("density", "2700[kg/m^3]");
    model.component("comp1").material("mat7").propertyGroup("Enu").set("youngsmodulus", "70e9[Pa]");
    model.component("comp1").material("mat7").propertyGroup("Enu").set("poissonsratio", "0.33");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("l", "");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("m", "");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("n", "");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("l", "-2.5e11[Pa]");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("m", "-3.3e11[Pa]");
    model.component("comp1").material("mat7").propertyGroup("Murnaghan").set("n", "-3.5e11[Pa]");
    model.component("comp1").material("mat7").propertyGroup("Lame").set("lambLame", "");
    model.component("comp1").material("mat7").propertyGroup("Lame").set("muLame", "");
    model.component("comp1").material("mat7").propertyGroup("Lame").set("lambLame", "5.1e10[Pa]");
    model.component("comp1").material("mat7").propertyGroup("Lame").set("muLame", "2.6e10[Pa]");

    /** Add physics - electromagnetics **/
    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").create("port1", "Port", 2);
    model.component("comp1").physics("emw").feature("port1").selection().set(1);
    model.component("comp1").physics("emw").create("sctr1", "Scattering", 2);
    model.component("comp1").physics("emw").feature("sctr1").selection().set(39, 46, 49, 56);
    model.component("comp1").physics("emw").feature("port1").set("Pin", "P_supply");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Rectangular");

    /** Add physics - heat transfer **/
    model.component("comp1").physics().create("ht", "HeatTransferInSolidsAndFluids", "geom1");
    model.component("comp1").physics("ht").selection().set(8, 9, 10, 11, 12, 13);
    model.component("comp1").physics("ht").feature("fluid1").selection().named("geom1_sel3");
    model.component("comp1").physics("ht").create("hs1", "HeatSource", 3);
    model.component("comp1").physics("ht").feature("hs1").selection().all();
    model.component("comp1").physics("ht").prop("PhysicalModelProperty").set("Tref", "T0");
    model.component("comp1").physics("ht").feature("init1").set("Tinit", "T0");
    model.component("comp1").physics("ht").feature("hs1").set("materialType", "from_mat");
    model.component("comp1").physics("ht").feature("hs1").set("Q0_src", "root.comp1.emw.Qh");

    /** Add multiphysics **/
    model.component("comp1").multiphysics().create("emh1", "ElectromagneticHeating", -1);

    /** Meshing setting **/
    model.component("comp1").mesh("mesh1").create("size1", "Size");
    model.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
    model.component("comp1").mesh("mesh1").feature("size1").selection().named("geom1_sel3");
    model.component("comp1").mesh("mesh1").feature("ftet1").selection().geom("geom1", 3);
    model.component("comp1").mesh("mesh1").feature("ftet1").selection().all();
    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);
    model.component("comp1").mesh("mesh1").feature("size1").set("hauto", 3);
    model.component("comp1").mesh("mesh1").feature("size1").set("table", "cfd");
    model.component("comp1").mesh("mesh1").run();

    /**Solver setting**/
    model.study().create("std1");
    model.study("std1").create("freq", "Frequency");
    model.study("std1").feature("freq")
         .set("activate", new String[]{"ht", "off", "emw", "on", "frame:spatial1", "on"});

    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").attach("std1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");

    model.study("std1").feature("freq").set("plist", "f_sup");

    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("v1").set("clistctrl", new String[]{"p1"});
    model.sol("sol1").feature("v1").set("cname", new String[]{"freq"});
    model.sol("sol1").feature("v1").set("clist", new String[]{"f_sup"});
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f_sup"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").set("itrestart", 300);
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("iter", 1);
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1")
         .set("linsolver", "pardiso");
    model.sol("sol1").runAll();

    /** Calculate power and output **/
    model.result().numerical().create("int1", "IntVolume");
    model.result().numerical("int1").selection().named("geom1_sel3");
    model.result().table().create("tbl1", "Table");
    model.result().numerical("int1").set("table", "tbl1");
    model.result().numerical("int1").set("expr", new String[]{"emw.Qe"});
    model.result().numerical("int1").set("unit", new String[]{"W"});
    model.result().numerical("int1").set("descr", new String[]{"Electromagnetic power loss density"});
    model.result().numerical("int1").setResult();
    model.result().table("tbl1").save("../Stub_tuner/simulation_result.csv");



    return model;
  }

  public static Model run2(Model model) {

    return model;
  }

  public static void main(String[] args) {
    Model model = run();
    run2(model);
  }

}
