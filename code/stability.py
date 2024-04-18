from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
import numpy as np
import os, sys

if __name__ == '__main__':


    if int(sys.argv[1]) == 0:
        aero_flag = False
    else:
        aero_flag = True
    if int(sys.argv[2]) == 0:
        tower_dt_dof = False
    else:
        tower_dt_dof = True
    if int(sys.argv[3]) == 0:
        controller = False
    else:
        controller = True
    if int(sys.argv[4]) == 0:
        unsteady_aero = False
    else:
        unsteady_aero = True
    if int(sys.argv[5]) == 0:
        pitch_sweep = False
    else:
        pitch_sweep = True
    if int(sys.argv[6]) == 0:
        BD = False
    else:
        BD = True
    
    if pitch_sweep==True and aero_flag==True:
        raise Exception('Please run a pitch sweep with the aero flag turned OFF')


    fastBatch = runFAST_pywrapper_batch()

    use_cores = 104

    run_dir = os.path.dirname( os.path.realpath(__file__))
    fastBatch.FAST_InputFile = 'IEA-15-240-RWT-Monopile.fst'
    fastBatch.FAST_directory = "/projects/windse/pbortolo/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile"

    folder_name = 'stability_iea15_'
    if aero_flag:
        folder_name += 'a_'
    else:
        folder_name += 's_'

    if tower_dt_dof:
        folder_name += 'towdt_'
    else:
        folder_name += 'rotor_'

    if controller:
        folder_name += 'ctrl_'
    else:
        folder_name += 'fixrpm_'

    if unsteady_aero:
        folder_name += 'ua'
    else:
        folder_name += 'sa'

    if pitch_sweep:
        folder_name += '_pitch'
    else:
        folder_name += '_rpm'
    if BD:
        folder_name += '_BD'
    else:
        folder_name += '_ED'



    fastBatch.FAST_runDirectory = os.path.join(run_dir, '..', '..', 'outputs',folder_name)
    
    if aero_flag:
        traj = np.loadtxt(os.path.join(run_dir, 'trajectory.dat'))

        hws_interp = traj[:,0]
        rot_speeds_interp = traj[:,1]
        pitch_interp = traj[:,2]
        # TTDspFA = traj[:,3]
        # TTDspSS = traj[:,4]
        Ct = traj[:,3]
        tau1_const_interp = np.zeros_like(hws_interp)
        for i in range(len(Ct)):
            a = 1. / 2. * (1. - np.sqrt(1. - Ct[i]))
            tau1_const_interp[i] = 1.1 / (1. - 1.3 * np.min([a, 0.5])) * 121. / hws_interp[i]
        # hws_interp = np.arange(3., 26., 1.)
        # rot_speeds_interp = np.interp(hws_interp, hws, rot_speeds)
        # pitch_interp = np.interp(hws_interp, hws, pitch)
        TTDspFA_interp = np.zeros_like(rot_speeds_interp)
        TTDspSS_interp = np.zeros_like(rot_speeds_interp)
        #tau1_const_interp = np.zeros_like(rot_speeds_interp)
    else:
        if pitch_sweep:
            pitch_interp = np.arange(-90, 100, 10)
            rot_speeds_interp = np.ones_like(pitch_interp) * 7.559987
        else:
            rot_speeds_interp = np.arange(0., 9., 1.)
            pitch_interp = np.zeros_like(rot_speeds_interp)
        TTDspFA_interp = np.zeros_like(rot_speeds_interp)
        TTDspSS_interp = np.zeros_like(rot_speeds_interp)
    
    n_cores = np.min([use_cores, len(rot_speeds_interp)])

    trim_case = np.zeros(len(rot_speeds_interp), dtype=int)
    trim_gain = np.zeros(len(rot_speeds_interp))
    trim_tol = np.zeros(len(rot_speeds_interp))
    VS_SlPc = np.zeros(len(rot_speeds_interp))
    VS_RtGnSp = np.zeros(len(rot_speeds_interp)) + 1.e-3
    for i in range(len(rot_speeds_interp)):
        trim_tol[i] = 0.0001
        VS_SlPc[i] = 2.
        if rot_speeds_interp[i] < max(rot_speeds_interp):
            trim_case[i] = 2
            trim_gain[i] = 100
            VS_RtGnSp[i] = max(rot_speeds_interp)
        else:
            trim_case[i] = 3
            trim_gain[i] = 0.00001


    if aero_flag:
        CompAero = np.ones_like(rot_speeds_interp, dtype=int) * 2
        CompInflow = np.ones_like(rot_speeds_interp, dtype=int)
        if hws_interp[0] == 0.:
            CompAero[0] = 0
            CompInflow[0] = 0
            rot_speeds_interp[0] = 0.
            pitch_interp[0] = 0.
            TTDspFA_interp[0] = 0.
            TTDspSS_interp[0] = 0.
    else:
        CompAero = np.zeros_like(rot_speeds_interp, dtype=int)
        CompInflow = np.zeros_like(rot_speeds_interp, dtype=int)

    case_inputs = {}
    case_inputs[("ElastoDyn","FlapDOF1")] = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")] = {'vals':["True"], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")] = {'vals':["True"], 'group':0}
    if controller:
        case_inputs[("ElastoDyn","GenDOF")] = {'vals':["True"], 'group':0}
    else:
        case_inputs[("ElastoDyn","GenDOF")] = {'vals':["False"], 'group':0}

    if tower_dt_dof:
        case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':["True"], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':["True"], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':["True"], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':["True"], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':["True"], 'group':0}
    else:
        case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':["False"], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':["False"], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':["False"], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':["False"], 'group':0}
        case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","TeetDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","YawDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","ShftTilt")] = {'vals':[0.], 'group':0}
    case_inputs[("ElastoDyn","PtfmSgDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmSwDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmHvDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmRDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmPDOF")] = {'vals':["False"], 'group':0}
    case_inputs[("ElastoDyn","PtfmYDOF")] = {'vals':["False"], 'group':0}

    case_inputs[("Fst","Gravity")] = {'vals':[0.], 'group':0}
    case_inputs[("Fst","Echo")] = {'vals':["False"], 'group':0}
    case_inputs[("Fst","TMax")] = {'vals':[2000.], 'group':0}
    if BD:
        case_inputs[("Fst","DT")] = {'vals':[0.0004], 'group':0}
        case_inputs[("Fst","CompElast")] = {'vals':[2], 'group':0}
    else:
        case_inputs[("Fst","DT")] = {'vals':[0.01], 'group':0}
        case_inputs[("Fst","CompElast")] = {'vals':[1], 'group':0}
    case_inputs[("Fst","InterpOrder")] = {'vals':[1], 'group':0}
    case_inputs[("Fst","CompAero")] = {'vals':CompAero, 'group':1}
    case_inputs[("Fst","CompInflow")] = {'vals':CompInflow, 'group':1}
    if controller:
        case_inputs[("Fst","CompServo")] = {'vals':[1], 'group':0}
    else:
        case_inputs[("Fst","CompServo")] = {'vals':[0], 'group':0}
    case_inputs[("Fst","CompHydro")] = {'vals':[0], 'group':0}
    case_inputs[("Fst","CompSub")] = {'vals':[0], 'group':0}
    case_inputs[("Fst","SumPrint")] = {'vals':["False"], 'group':0}
    case_inputs[("Fst","DT_Out")] = {'vals':[0.02], 'group':0}
    case_inputs[("Fst","TStart")] = {'vals':[0.0], 'group':0}
    case_inputs[("Fst","OutFmt")] = {'vals':["ES17.9E3"], 'group':0}
    case_inputs[("Fst","OutFileFmt")] = {'vals':[2], 'group':0}
    case_inputs[("Fst","Linearize")] = {'vals':["True"], 'group':0}
    case_inputs[("Fst","CalcSteady")] = {'vals':["True"], 'group':0}
    case_inputs[("Fst","TrimCase")] = {'vals':trim_case, 'group':1}
    case_inputs[("Fst","TrimTol")] = {'vals':trim_tol, 'group':1}
    case_inputs[("Fst","TrimGain")] = {'vals':trim_gain, 'group':1}
    case_inputs[("Fst","Twr_Kdmp")] = {'vals':[2.e+3], 'group':0}
    case_inputs[("Fst","Bld_Kdmp")] = {'vals':[2.e+3], 'group':0}
    case_inputs[("Fst","NLinTimes")] = {'vals':[2], 'group':0}
    case_inputs[("Fst","LinInputs")] = {'vals':[0], 'group':0}
    case_inputs[("Fst","LinOutputs")] = {'vals':[0], 'group':0}
    case_inputs[("Fst","LinOutJac")] = {'vals':["False"], 'group':0}
    case_inputs[("Fst","LinOutMod")] = {'vals':["False"], 'group':0}
    case_inputs[("Fst","WrVTK")] = {'vals':[3], 'group':0}
    case_inputs[("Fst","VTK_type")] = {'vals':[1], 'group':0}
    case_inputs[("Fst","VTK_fields")] = {'vals':["False"], 'group':0}
    case_inputs[("Fst","VTK_fps")] = {'vals':[15], 'group':0}
    if aero_flag:
        if unsteady_aero:
            case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':[2], 'group':0}
            case_inputs[("AeroDyn15","UAMod")] = {'vals':[4], 'group':0}
            case_inputs[("AeroDyn15","WakeMod")] = {'vals':[12], 'group':0}
            case_inputs[("AeroDyn15","DBEMT_Mod")] = {'vals':[3], 'group':0}
            case_inputs[("AeroDyn15","tau1_const")] = {'vals':tau1_const_interp, 'group':1}

        else:
            case_inputs[("AeroDyn15","WakeMod")] = {'vals':[1], 'group':0}
            case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':[1], 'group':0}
        case_inputs[("AeroDyn15","TwrPotent")] = {'vals':[0], 'group':0}
        case_inputs[("AeroDyn15","TwrShadow")] = {'vals':[0], 'group':0}
        case_inputs[("AeroDyn15","FrozenWake")] = {'vals':["True"], 'group':0}
        case_inputs[("AeroDyn15","SkewMod")] = {'vals':[0], 'group':0}
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        case_inputs[("InflowWind","HWindSpeed")]= {'vals': hws_interp, 'group': 1}
        case_inputs[("InflowWind","PLexp")] = {'vals':[0.], 'group':0}
    case_inputs[("ElastoDyn","RotSpeed")] = {'vals': rot_speeds_interp, 'group': 1}
    case_inputs[("ElastoDyn","BlPitch1")] = {'vals': pitch_interp, 'group': 1}
    case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","TTDspFA")] = {'vals': TTDspFA_interp, 'group': 1}
    case_inputs[("ElastoDyn","TTDspSS")] = {'vals': TTDspSS_interp, 'group': 1}
    if controller:
        case_inputs[("ServoDyn","PCMode")] = {'vals': [0], 'group': 0}
        case_inputs[("ServoDyn","VSContrl")] = {'vals': [1], 'group': 0}
        case_inputs[("ServoDyn","VS_SlPc")] = {'vals': VS_SlPc, 'group': 1}
        case_inputs[("ServoDyn","VS_RtGnSp")] = {'vals': VS_RtGnSp, 'group': 1}

    namebase = 'iea15_stab'

    case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=fastBatch.FAST_runDirectory, namebase=namebase)

    fastBatch.case_list = case_list
    fastBatch.case_name_list = case_name_list
    fastBatch.use_exe = True
    fastBatch.FAST_exe = '/home/pbortolo/openfast/openfast_rc3p5p3/build/glue-codes/openfast/openfast'

    if n_cores > 1:
        fastBatch.run_multi(n_cores)
    else:
        fastBatch.run_serial()
