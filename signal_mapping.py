#!/usr/bin/env python3
"""
Comprehensive Signal Mapping Dictionary for Multi-OEM Vehicle Data Analysis

This module provides a unified, robust signal mapping system that supports:
- Multiple OEMs (BMW, VW/Audi, Mercedes, Ford, GM, Toyota, Honda, Nissan, etc.)
- Different measurement systems (ETAS, Vector, INCA, MDF, CAN)
- Various naming conventions (English, German, Japanese)
- Case-insensitive and fuzzy matching

Author: OEM Calibration Engineering Team
Version: 2.0-OEM
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import re

# ==============================================================================
# COMPREHENSIVE SIGNAL MAPPING DICTIONARY
# ==============================================================================
# Each signal role maps to a list of possible names found across different
# OEMs, measurement tools, and vehicle platforms
# ==============================================================================

SIGNAL_MAP: Dict[str, List[str]] = {
    # ========================================================================
    # ENGINE SPEED / RPM
    # ========================================================================
    "rpm": [
        # Generic/Common
        "rpm", "RPM", "EngineSpeed", "Engine_Speed", "EngineSpeedActual", 
        "EngSpeed", "nEng", "n_Eng", "EngineRPM", "Engine_RPM",
        "CrankSpeed", "CrankshaftSpeed", "Crankshaft_RPM", "nCrank", "CrankRPM",
        "Ext_nEng", "nEng_Act", "RpmEng", "EngRPM", "EngineSpd",
        
        # OBD-II PIDs (Mode 01)
        "PID_0C", "PID_0C_RPM", "EngineSpeed_PID", "RPM_PID", "Engine_RPM_PID",
        
        # ETAS/Vector naming
        "Epm_nEng", "Epm_nEng_RTE", "Ext_nEng_RTE", "nEng_Ext",
        "inRpm", "outRpmSpeed", "inRpmSpeed2", "RpmEngActual",
        
        # VW/Audi/SEAT/Skoda
        "Motordrehzahl", "n_Mot", "nMot", "Drehzahl", "n_Drehzahl",
        "Mot_Drehzahl", "MotDrehzahl", "n_Engine", "Engine_n",
        "VKM_nMot", "GKM_nMot",
        
        # BMW
        "Motordrehzahl", "n_Motor", "MotorDrehzahl", "MotDrehzahl",
        "GetriebeEingang", "nMotor",
        
        # Mercedes-Benz
        "Motordrehzahl", "n_Motor", "Motor_n", "Drehzahl_Motor",
        "N_MOT", "N_MOTOR",
        
        # Ford
        "EngineSpeed", "RPM_Engine", "Engine_RPM_Actual", "RPM_Act",
        "N_Engine", "Engine_N",
        
        # GM/Opel
        "EngineRPM", "RPM", "Engine_Speed", "Speed_Engine",
        "RPM_Actual", "Eng_RPM",
        
        # Toyota/Lexus
        "NE", "NE_ENG", "Engine_NE", "RPM_Engine", "RPM",
        
        # Honda/Acura
        "RPM", "EngineRPM", "ENG_SPEED", "Engine_Speed",
        
        # Nissan/Infiniti
        "RPM", "EngineRPM", "ENG_SPD", "Engine_Speed",
        
        # Hyundai/Kia
        "EngineSpeed", "RPM", "ENG_RPM", "Engine_RPM",
        
        # Fiat/Chrysler
        "EngineSpeed", "RPM_Engine", "EngineRPM",
        
        # PSA (Peugeot/Citroen)
        "Vitesse_Moteur", "VM", "n_Moteur", "Moteur_Vitesse",
        
        # Renault/Nissan
        "Vitesse_Moteur", "VM", "RPM_Moteur",
        
        # Chinese OEMs (BYD, Geely, Great Wall)
        "EngineSpeed", "RPM", "发动机转速", "Engine_RPM",
        
        # Indian OEMs (Tata, Mahindra)
        "EngineSpeed", "RPM", "Engine_RPM",
    ],
    
    # ========================================================================
    # ENGINE TORQUE / LOAD
    # ========================================================================
    "torque": [
        # Generic/Common
        "Torque", "EngineTorque", "Engine_Torque", "Tq", "Trq", "tqEng",
        "Trq_Ext", "TqSys_tqCkEngReal", "EngineLoad", "Load", "LoadPercent",
        "EngineLoadPercent", "EngineTorqueActual", "Torque_Act", "TqEng",
        "Engine_Torque_Actual", "TorqueValue", "EngineTorqueValue",
        
        # OBD-II PIDs (Mode 01)
        "PID_0B", "PID_0B_Load", "EngineLoad_PID", "Torque_PID",
        
        # ETAS/Vector
        "TqSys_tqCkEngReal_RTE", "inTorque", "trqEng", "Tq_Engine",
        
        # VW/Audi
        "Motormoment", "Drehmoment", "T_Mot", "t_Mot", "M_Mot",
        "Motormoment_Soll", "Motormoment_Ist", "T_Motor",
        "VKM_tMot", "GKM_tMot",
        
        # BMW
        "Motormoment", "TorqueMotor", "M_Motor", "T_Motor",
        "Mot_Moment", "MotorMoment",
        
        # Mercedes-Benz
        "Motormoment", "Torque_Motor", "M_MOT", "T_MOTOR",
        "MotorMoment", "Moment_Motor",
        
        # Ford
        "Engine_Torque", "Torque_Engine", "EngineTorque",
        "Torque_Actual", "Tq_Eng",
        
        # GM
        "EngineTorque", "Torque", "TorqueValue", "Engine_Tq",
        
        # Toyota
        "TRQ", "TRQ_ENG", "Engine_Torque", "Torque",
        
        # Honda
        "TORQUE", "ENG_TORQUE", "EngineTorque",
        
        # Nissan
        "TRQ", "EngineTorque", "ENG_TORQUE",
        
        # Load-based
        "EngineLoad", "Load", "LoadPercent", "EngLoad", "Load_Eng",
        "BMEP", "BrakeMeanEffectivePressure", "Load_Percent",
    ],
    
    # ========================================================================
    # LAMBDA / AIR-FUEL RATIO
    # ========================================================================
    "lambda": [
        # Generic/Common
        "Lambda", "lambda", "AFR", "AirFuelRatio", "Air_Fuel_Ratio", "afr",
        "Lambda_B1", "Lambda_B2", "Lambda1", "Lambda2", "Lam", "O2Sensor",
        "OxygenSensor", "WidebandLambda", "LambdaSensor", "AirFuel",
        "StoichRatio", "Lambda_Act", "Lambda_Meas",
        
        # ETAS/Vector
        "ExM_Lam_Estim", "ExM_Lam_Actual", "Lambda_Exh", "Lam_Bank1", "Lam_Bank2",
        
        # VW/Audi
        "Lambda", "Lambda_Wert", "LAM", "LAM_Wert", "Lam_1", "Lam_2",
        "Lambda_B1", "Lambda_B2", "AFR", "Luft_Kraftstoff",
        "VKM_Lam", "GKM_Lam",
        
        # BMW
        "Lambda", "LAM", "Lambda_Wert", "AFR", "Luft_Kraftstoff_Verhaeltnis",
        
        # Mercedes-Benz
        "Lambda", "LAM", "Lambda_Wert", "AFR",
        
        # Ford
        "Lambda", "AFR", "AirFuelRatio", "Lambda_Bank1", "Lambda_Bank2",
        
        # GM
        "Lambda", "AFR", "AirFuelRatio", "EQ_RATIO", "EquivalenceRatio",
        
        # Toyota
        "LAMBDA", "AFR", "Lambda_Value", "EQ_RATIO",
        
        # Honda
        "LAMBDA", "AFR", "Lambda", "O2_Sensor",
        
        # Nissan
        "LAMBDA", "AFR", "Lambda", "O2_SENSOR",
        
        # Oxygen sensors
        "O2Sensor", "O2_B1S1", "O2_B1S2", "O2_B2S1", "O2_B2S2",
        "OxygenSensor", "O2_Voltage", "O2_Current", "Wideband",
    ],
    
    # ========================================================================
    # COOLANT TEMPERATURE
    # ========================================================================
    "coolant_temp": [
        # Generic/Common
        "CoolantTemp", "ECT_C", "ECT", "EngineTemp", "Coolant_Temp",
        "temp_coolant", "CoolTemp", "EngineCoolantTemp", "T_Coolant",
        "Twat", "TCO", "CoolantTemperature", "TempCoolant",
        
        # OBD-II PIDs (Mode 01)
        "PID_05", "PID_05_ECT", "ECT_PID", "CoolantTemp_PID",
        
        # VW/Audi
        "Kuehlmitteltemperatur", "T_Kuehlmittel", "TCO", "T_CO",
        "Kuehlmittel_Temp", "VKM_TCO", "GKM_TCO",
        
        # BMW
        "Kuehlmitteltemperatur", "T_Kuehlmittel", "TCO", "Kuehl_Temp",
        
        # Mercedes-Benz
        "Kuehlmitteltemperatur", "T_KUEHL", "TCO",
        
        # Ford
        "ECT", "CoolantTemp", "EngineCoolantTemp", "T_CO",
        
        # GM
        "ECT", "CoolantTemp", "EngineCoolantTemp",
        
        # Toyota
        "THW", "ECT", "CoolantTemp", "Engine_Coolant_Temp",
        
        # Honda
        "ECT", "COOLANT_TEMP", "EngineCoolantTemp",
        
        # Nissan
        "ECT", "COOLANT_TEMP", "EngineCoolantTemp",
    ],
    
    # ========================================================================
    # INTAKE AIR TEMPERATURE
    # ========================================================================
    "intake_air_temp": [
        # Generic/Common
        "IAT_C", "IntakeAirTemp", "IAT", "IntakeTemp", "AirInletTemp",
        "Temp_AirInlet", "IATemp", "T_AirInlet", "IntakeAirTemperature",
        "AirTemp", "InletAirTemp",
        
        # VW/Audi
        "Ansauglufttemperatur", "T_Ansaug", "TAT", "T_AT",
        "Ansaug_Temp", "VKM_TAT", "GKM_TAT",
        
        # BMW
        "Ansauglufttemperatur", "T_Ansaug", "TAT",
        
        # Mercedes-Benz
        "Ansauglufttemperatur", "T_ANSAUG", "TAT",
        
        # Ford
        "IAT", "IntakeAirTemp", "IAT_C",
        
        # GM
        "IAT", "IntakeAirTemp", "IAT_C",
        
        # Toyota
        "THA", "IAT", "IntakeAirTemp",
        
        # Honda
        "IAT", "INTAKE_AIR_TEMP", "IntakeAirTemp",
        
        # Nissan
        "IAT", "INTAKE_AIR_TEMP", "IntakeAirTemp",
    ],
    
    # ========================================================================
    # CRANKSHAFT ANGLE / POSITION
    # ========================================================================
    "crank_angle": [
        # Generic/Common
        "CrankAngle", "CrankPos", "CrankshaftAngle", "CrankPosition", "CrankAng",
        "CrankshaftPos", "CrankAngleDeg", "Angle", "Pos_Crank", "nCrankAng",
        "CKP_Angle", "CrankAngPos", "Crank_Deg", "Phi_Crank", "CrankPos_Deg",
        "CrankAngle_Actual", "CrankAngle_Meas",
        
        # VW/Audi
        "Kurbelwellenwinkel", "KW_Winkel", "CrankAngle", "KW_Angle",
        "Kurbelwellenposition", "KW_Pos",
        
        # BMW
        "Kurbelwellenwinkel", "KW_Winkel", "CrankAngle",
        
        # Mercedes-Benz
        "Kurbelwellenwinkel", "KW_WINKEL", "CrankAngle",
        
        # Ford
        "CrankAngle", "CrankPosition", "CKP_Angle",
        
        # GM
        "CrankAngle", "CrankPosition", "CKP_Angle",
        
        # Toyota
        "CRK_ANG", "CrankAngle", "CrankPosition",
        
        # Honda
        "CRANK_ANGLE", "CrankAngle", "CKP_ANGLE",
        
        # Nissan
        "CRANK_ANGLE", "CrankAngle", "CKP_ANGLE",
    ],
    
    # ========================================================================
    # IGNITION TIMING / SPARK ADVANCE
    # ========================================================================
    "ignition_timing": [
        # Generic/Common
        "IgnitionTiming", "IgnTiming", "SparkAdvance", "IgnAngle", "IgnAdvance",
        "IgnAng", "SparkTiming", "IgnAdv", "SparkAdv", "IgnitionAngle",
        "SparkAngle", "IgnitionAdvance", "Ign_Angle", "Ign_Adv",
        
        # VW/Audi
        "Zuendzeitpunkt", "ZW", "ZP", "Zuend_Winkel", "Z_Winkel",
        "VKM_ZW", "GKM_ZW",
        
        # BMW
        "Zuendzeitpunkt", "ZW", "Zuend_Winkel",
        
        # Mercedes-Benz
        "Zuendzeitpunkt", "ZW", "ZUEND_WINKEL",
        
        # Ford
        "IgnitionTiming", "SparkAdvance", "IgnAngle",
        
        # GM
        "IgnitionTiming", "SparkAdvance", "IgnAngle",
        
        # Toyota
        "IGT", "IgnitionTiming", "SparkAdvance",
        
        # Honda
        "IGNITION_TIMING", "IgnitionTiming", "SparkAdvance",
        
        # Nissan
        "IGNITION_TIMING", "IgnitionTiming", "SparkAdvance",
    ],
    
    # ========================================================================
    # MANIFOLD ABSOLUTE PRESSURE (MAP) / BOOST
    # ========================================================================
    "map_sensor": [
        # Generic/Common
        "MAP", "MAP_kPa", "ManifoldPressure", "Manifold_Abs_Pressure", "Boost",
        "Boost_kPa", "MAP_sensor", "P_MAP", "Pressure_Intake", "PIntake",
        "IntakeManifoldPressure", "MAP_Pressure", "Boost_Pressure",
        "IntakePressure", "Manifold_Pressure",
        
        # VW/Audi
        "Ladedruck", "LD", "Ladedruck_Absolut", "Boost", "Ladedruck_Soll",
        "VKM_LD", "GKM_LD",
        
        # BMW
        "Ladedruck", "LD", "Boost", "Ladedruck_Absolut",
        
        # Mercedes-Benz
        "Ladedruck", "LD", "BOOST",
        
        # Ford
        "MAP", "ManifoldPressure", "Boost_Pressure",
        
        # GM
        "MAP", "ManifoldPressure", "MAP_kPa",
        
        # Toyota
        "MAP", "MAP_kPa", "ManifoldPressure",
        
        # Honda
        "MAP", "MAP_kPa", "ManifoldPressure",
        
        # Nissan
        "MAP", "MAP_kPa", "ManifoldPressure",
    ],
    
    # ========================================================================
    # THROTTLE POSITION / ACCELERATOR PEDAL
    # ========================================================================
    "throttle": [
        # Generic/Common
        "ThrottlePos", "ThrottlePosition", "AccPedalPos", "PedalPos",
        "Throttle", "APP", "TPS", "ThrottleAct", "AccPedal", "Throttle_Act",
        "ThrottleAngle", "Throttle_Position", "APP_Pos", "Pedal_Position",
        
        # OBD-II PIDs (Mode 01)
        "PID_11", "PID_11_Throttle", "ThrottlePosition_PID", "TPS_PID",
        
        # VW/Audi
        "Drosselklappenposition", "DK_Pos", "DKP", "Drosselklappe",
        "Gaspedalposition", "GPP", "Pedal_Pos", "VKM_DKP", "GKM_DKP",
        
        # BMW
        "Drosselklappenposition", "DK_Pos", "DKP", "Gaspedal",
        
        # Mercedes-Benz
        "Drosselklappenposition", "DK_POS", "DKP", "Gaspedal",
        
        # Ford
        "ThrottlePosition", "APP", "AcceleratorPedalPosition",
        
        # GM
        "ThrottlePosition", "TPS", "ThrottlePos",
        
        # Toyota
        "TH", "ThrottlePosition", "Throttle_Pos",
        
        # Honda
        "THROTTLE_POS", "ThrottlePosition", "TPS",
        
        # Nissan
        "THROTTLE_POS", "ThrottlePosition", "TPS",
    ],
    
    # ========================================================================
    # CYLINDER COUNT
    # ========================================================================
    "cylinder_count": [
        "nCyl", "CylinderCount", "NumCylinders", "EngineCylinders", "Cylinders",
        "nCylinders", "CylCount", "EngineCyl", "NumberOfCylinders",
        "CylindersCount", "Cyl_Count",
    ],
    
    # ========================================================================
    # FUEL CONSUMPTION / FUEL RATE
    # ========================================================================
    "fuel_rate": [
        # Generic/Common
        "FuelRate", "fuel_flow", "FuelCons", "FuCns_volFuCnsTot", "FuelConsumption",
        "FuelFlow", "Fuel_Flow", "FuelRate_LH", "FuelRate_L_Min",
        "FuelMassFlow", "FuelVolFlow", "Fuel_Rate", "FuelConsumptionRate",
        "FuelFlowRate", "Fuel_Mass_Flow", "Fuel_Vol_Flow",
        
        # OBD-II PIDs (Mode 01)
        "PID_2F", "PID_2F_Fuel", "FuelTankLevelInput", "Fuel_Rail_Pressure", "FuelRailPressure",
        "Fuel_Pressure", "FuelPressure", "FRP", "Fuel_Tank_Level",
        
        # VW/Audi
        "Kraftstoffverbrauch", "KFZ_Verbrauch", "FuelCons", "Kraftstoffdurchfluss",
        "FuelFlowRate", "VKM_Fuel", "GKM_Fuel",
        
        # BMW
        "Kraftstoffverbrauch", "FuelConsumption", "Kraftstoffdurchfluss",
        
        # Mercedes-Benz
        "Kraftstoffverbrauch", "FuelConsumption", "KRAFTSTOFFVERBRAUCH",
        
        # Ford
        "FuelRate", "FuelConsumption", "FuelFlow", "Fuel_Rate_Actual",
        
        # GM
        "FuelRate", "FuelConsumption", "FuelFlow", "FUEL_FLOW_RATE",
        
        # Toyota
        "FUEL_RATE", "FuelConsumption", "FuelFlow", "FUEL_FLOW",
        
        # Honda
        "FUEL_RATE", "FuelConsumption", "FuelFlow",
        
        # Nissan
        "FUEL_RATE", "FuelConsumption", "FuelFlow",
        
        # Units variations
        "FuelCons_L_H", "FuelCons_L_100km", "FuelCons_gps", "FuelCons_kg_h",
        "FuelFlow_Lh", "FuelFlow_L_h", "FuelFlow_g_s", "FuelFlow_kg_s",
        "FuelRate_LPH", "FuelRate_LPH", "FuelRate_gph", "FuelRate_g_h",
        
        # CAN/DBC variations
        "CAN_FuelRate", "CAN_FuelFlow", "DBC_FuelRate", "FuelRate_CAN",
        "FuelCons_CAN", "FuelFlow_CAN", "FR_CAN",
    ],
    
    # ========================================================================
    # AIR MASS FLOW
    # ========================================================================
    "air_mass_flow": [
        # Generic/Common
        "air_mass_flow", "MAF_gps", "InM_mfAirCanPurgEstim", "mAir", "AirFlow",
        "MAF", "MassAirFlow", "AirMassFlow", "MAF_g_s", "AirFlowRate",
        "Air_Mass_Flow", "MAF_Rate", "AirFlow_Rate", "AirMassFlowRate",
        
        # OBD-II PIDs (Mode 01)
        "PID_10", "PID_10_MAF", "MassAirFlow", "MAF_Value", "MAF_Sensor",
        
        # VW/Audi
        "Luftmasse", "Luftmassenstrom", "m_Luft", "AirMass", "MAF",
        "VKM_MAF", "GKM_MAF",
        
        # BMW
        "Luftmasse", "Luftmassenstrom", "MAF", "AirMass",
        
        # Mercedes-Benz
        "Luftmasse", "Luftmassenstrom", "MAF", "AIR_MASS",
        
        # Ford
        "MAF", "MassAirFlow", "AirMassFlow", "MAF_Sensor",
        
        # GM
        "MAF", "MassAirFlow", "AirMassFlow", "MAF_GM",
        
        # Toyota
        "MAF", "MAF_VOL", "MassAirFlow", "AirMassFlow",
        
        # Honda
        "MAF", "MassAirFlow", "AirMassFlow",
        
        # Nissan
        "MAF", "MassAirFlow", "AirMassFlow",
        
        # Units variations
        "MAF_g_s", "MAF_gps", "MAF_kg_h", "MAF_lb_min", "MAF_lb_h",
        "AirFlow_g_s", "AirFlow_kg_h", "AirFlow_lb_min",
        "mAir_g_s", "mAir_kg_h", "mAir_lb_min",
        
        # CAN/DBC variations
        "CAN_MAF", "CAN_AirMassFlow", "DBC_MAF", "MAF_CAN",
        "AirMass_CAN", "AirFlow_CAN", "MAF_DBC",
    ],
    
    # ========================================================================
    # EXHAUST TEMPERATURE
    # ========================================================================
    "exhaust_temp": [
        "ExM_tExMnEstim_RTE", "exhaust_temp", "TExh", "EGT", "ExhTemp",
        "ExhaustTemp", "EGT_Temp", "ExhaustGasTemp", "T_Exh",
    ],
    
    # ========================================================================
    # OIL TEMPERATURE
    # ========================================================================
    "oil_temp": [
        "OilTemp", "temp_oil", "Oil_Temp", "EngineOilTemp", "OilTemperature",
        "T_Oil", "TOil",
    ],
    
    # ========================================================================
    # BATTERY VOLTAGE
    # ========================================================================
    "battery_voltage": [
        "battery_voltage", "VBatt", "u_batt", "BatteryVoltage", "VBat",
        "BattVolt", "Battery_Voltage", "U_Batt", "V_Batt",
    ],
    
    # ========================================================================
    # VEHICLE SPEED (for gear hunting)
    # ========================================================================
    "vehicle_speed": [
        # Generic/Common
        "Veh_spdVeh", "Ext_spdVeh", "Vehicle_Speed", "VehSpd", "v", "VehicleSpeed",
        "Speed", "VehSpeed", "SpdVeh", "vVeh", "VSS", "VehicleSpd", "Vehicle_Speed_Act",
        "VehSpd_Act", "VehicleSpeedActual", "SpdVeh_Act", "VehSpdV", "VSpeed",
        "v_Fahrzeug", "Fahrzeuggeschwindigkeit", "v_Speed",
        "Vehicle_Speed_Actual", "Veh_Speed", "VehicleSpeedVSS", "SpeedVSS",
        "SPEED", "VehicleSpeedValue", "SpeedValue",
        
        # OBD-II PIDs (Mode 01)
        "PID_0D", "PID_0D_Speed", "VehicleSpeed", "VSS", "Speed_KPH", "Speed_MPH",
        
        # VW/Audi variations
        "VITESSE_VEHICULE_ROUES", "CAN_VITESSE_VEHICULE_ROUES",
        "96D7124080_8128328U_FM77_nc_CAN_VITESSE_VEHICULE_ROUES",
        "Fahrzeuggeschwindigkeit", "Geschwindigkeit", "v_Fzg", "v_Fahrzeug",
        "VKM_VehSpd", "GKM_VehSpd",
        
        # BMW variations
        "Fahrzeuggeschwindigkeit", "Geschwindigkeit", "v_Fahrzeug",
        
        # Mercedes-Benz variations
        "Fahrzeuggeschwindigkeit", "Geschwindigkeit", "V_FZG",
        
        # Ford variations
        "VSS", "VehicleSpeedSensor", "VSS_Signal", "Vehicle_VSS",
        
        # GM variations
        "VSS", "VehicleSpeedSensor", "Vehicle_Speed_Sensor",
        
        # Toyota variations
        "VSS", "VehicleSpeed", "VEHICLE_SPEED", "SPD",
        
        # Honda variations
        "VSS", "VehicleSpeed", "VEHICLE_SPEED",
        
        # Nissan variations
        "VSS", "VehicleSpeed", "VEHICLE_SPEED",
        
        # CAN bus variations (different naming patterns)
        "CAN_Speed", "CAN_VSS", "CAN_VehicleSpeed", "DBC_Speed",
        "Speed_CAN", "VSS_CAN", "VehicleSpeed_CAN",
        
        # Units variations
        "Speed_kmh", "Speed_km_h", "Speed_KPH", "Speed_kph",
        "Speed_mph", "Speed_MPH", "Speed_m_s", "Speed_ms",
        "VehSpd_kmh", "VehSpd_KPH", "VehicleSpeed_kmh",
        
        # Alternative naming
        "v", "V", "vel", "velocity", "Velocity", "VEH_VEL",
        "spd", "SPD", "Spd", "speed_value", "Speed_Value",
    ],
    
    # ========================================================================
    # GEAR POSITION (for gear hunting)
    # ========================================================================
    "gear": [
        "Gear", "TrnsGr", "VSCtl_noGear", "Gearbox_Gear", "iGear", "GearSelected",
        "GearCurrent", "GearAct", "ActualGear", "GearPos", "nGear", "Gear_Act",
        "TransmissionGear", "TCU_Gear", "Gear_Actual", "GearValue", "CurrentGear",
        "GearPosition", "GearPos_Act", "TransGear", "Gearbox_Gear_Act",
        "ECM_Gear", "TCM_Gear", "TCM_GearActual", "Gear_Ratio", "GearRatio",
        "Gang_Position", "Gang", "Gear_Gang", "GearboxGear", "GetriebeGang",
        "GEAR", "GearValue", "TransmissionGearPosition",
    ],
    
    # ========================================================================
    # DISTANCE / ODOMETER (for fuel consumption calculations)
    # ========================================================================
    "distance": [
        # Generic/Common
        "Distance", "Odometer", "TotalDistance", "TripDistance", "Distance_Trip",
        "Dist", "Dist_Total", "Odo", "Odometer_Value", "Distance_Value",
        "Total_Distance", "Trip_Distance", "Dist_Trip", "Distance_Traveled",
        
        # OBD-II PIDs (Mode 01)
        "PID_31", "DistanceTraveled", "Distance_km", "Distance_miles",
        
        # VW/Audi
        "Fahrstrecke", "Kilometerstand", "KmStand", "Distance", "Strecke",
        "VKM_Distance", "GKM_Distance",
        
        # BMW
        "Fahrstrecke", "Kilometerstand", "Distance",
        
        # Mercedes-Benz
        "Fahrstrecke", "Kilometerstand", "DISTANCE",
        
        # Ford
        "Distance", "Odometer", "TripDistance", "Distance_Traveled",
        
        # GM
        "Distance", "Odometer", "Odometer_Reading",
        
        # Toyota
        "DISTANCE", "Odometer", "Trip_Distance",
        
        # Honda
        "DISTANCE", "Odometer", "Trip_Distance",
        
        # Nissan
        "DISTANCE", "Odometer", "Trip_Distance",
        
        # Units variations
        "Distance_km", "Distance_Km", "Distance_KM", "Distance_kmh",
        "Distance_miles", "Distance_Miles", "Distance_MI",
        "Odo_km", "Odo_miles", "Odometer_km", "Odometer_miles",
        
        # CAN/DBC variations
        "CAN_Distance", "CAN_Odometer", "DBC_Distance", "Distance_CAN",
        "Odometer_CAN", "Distance_DBC",
    ],
}

# Signal roles grouped by category for easier maintenance
CRITICAL_SIGNALS: Set[str] = {"rpm", "torque"}  # Always required
ENGINE_SIGNALS: Set[str] = {"rpm", "torque", "lambda", "ignition_timing", "coolant_temp", "intake_air_temp"}
DIAGNOSTIC_SIGNALS: Set[str] = {"crank_angle", "lambda", "coolant_temp", "throttle", "map_sensor"}
OPTIONAL_SIGNALS: Set[str] = {"fuel_rate", "air_mass_flow", "exhaust_temp", "oil_temp", "battery_voltage"}


# ==============================================================================
# ADVANCED SIGNAL FINDING WITH FUZZY MATCHING
# ==============================================================================

def find_signal_advanced(
    available_channels: List[str],
    signal_role: str,
    case_sensitive: bool = False,
    fuzzy_match: bool = True,
    substring_match: bool = True
) -> Optional[str]:
    """
    Advanced signal finding with multiple matching strategies.
    
    Args:
        available_channels: List of all available channel names in the MDF file
        signal_role: The role to search for (e.g., "rpm", "lambda", "torque")
        case_sensitive: Whether matching should be case-sensitive
        fuzzy_match: Enable fuzzy/partial matching
        substring_match: Enable substring matching
        
    Returns:
        First matching channel name or None
    """
    if signal_role not in SIGNAL_MAP:
        return None
    
    candidates = SIGNAL_MAP[signal_role]
    
    # Normalize channel names based on case sensitivity
    if case_sensitive:
        channel_map = {ch: ch for ch in available_channels}
        candidate_list = candidates
    else:
        channel_map = {ch.lower(): ch for ch in available_channels}
        candidate_list = [c.lower() for c in candidates]
    
    # Strategy 1: Exact match
    for cand in candidate_list:
        if cand in channel_map:
            return channel_map[cand]
    
    # Strategy 2: Substring match (candidate in channel or channel in candidate)
    if substring_match:
        for cand in candidate_list:
            for ch_lower, ch_original in channel_map.items():
                if cand in ch_lower or ch_lower in cand:
                    return ch_original
    
    # Strategy 3: Fuzzy match (word boundary matching)
    if fuzzy_match:
        # Split candidates into key words
        for cand in candidate_list:
            cand_words = re.split(r'[_\s-]+', cand)
            for ch_lower, ch_original in channel_map.items():
                ch_words = re.split(r'[_\s-]+', ch_lower)
                # Check if key words from candidate appear in channel
                if len(cand_words) > 0:
                    match_count = sum(1 for cw in cand_words if any(cw in cw2 for cw2 in ch_words))
                    if match_count >= len(cand_words) * 0.6:  # 60% word match threshold
                        return ch_original
    
    return None


def find_multiple_signals(
    available_channels: List[str],
    signal_roles: List[str],
    **kwargs
) -> Dict[str, Optional[str]]:
    """
    Find multiple signals at once.
    
    Returns:
        Dictionary mapping signal_role -> channel_name (or None if not found)
    """
    return {
        role: find_signal_advanced(available_channels, role, **kwargs)
        for role in signal_roles
    }


def get_all_candidates(signal_role: str) -> List[str]:
    """Get all candidate names for a signal role."""
    return SIGNAL_MAP.get(signal_role, [])


def get_signal_statistics(mdf_channels: List[str]) -> Dict[str, Any]:
    """
    Analyze which signals are available in an MDF file.
    
    Returns:
        Dictionary with statistics about signal availability
    """
    found_signals = {}
    missing_signals = {}
    
    for role, candidates in SIGNAL_MAP.items():
        found = find_signal_advanced(mdf_channels, role)
        if found:
            found_signals[role] = found
        else:
            missing_signals[role] = len(candidates)
    
    return {
        "found": found_signals,
        "missing": missing_signals,
        "found_count": len(found_signals),
        "total_roles": len(SIGNAL_MAP),
        "coverage_percent": round(len(found_signals) / len(SIGNAL_MAP) * 100, 1)
    }


# Backward compatibility aliases
RPM_CANDIDATES = SIGNAL_MAP["rpm"]
TORQUE_CANDIDATES = SIGNAL_MAP["torque"]
LAMBDA_CANDIDATES = SIGNAL_MAP["lambda"]
COOLANT_TEMP_CANDIDATES = SIGNAL_MAP["coolant_temp"]
INTAKE_TEMP_CANDIDATES = SIGNAL_MAP["intake_air_temp"]
CRANKSHAFT_ANGLE_CANDIDATES = SIGNAL_MAP["crank_angle"]
IGNITION_CANDIDATES = SIGNAL_MAP["ignition_timing"]
MAP_CANDIDATES = SIGNAL_MAP["map_sensor"]
THROTTLE_CANDIDATES = SIGNAL_MAP["throttle"]
CYLINDER_COUNT_CANDIDATES = SIGNAL_MAP["cylinder_count"]


def find_signal_in_dataframe(df_columns: List[str], signal_role: str) -> Optional[str]:
    """
    Find signal in DataFrame columns using signal mapping system.
    Works with CSV/Excel files by matching column names.
    
    Args:
        df_columns: List of column names from pandas DataFrame
        signal_role: The role to search for (e.g., "rpm", "lambda", "torque")
        
    Returns:
        First matching column name or None
    """
    return find_signal_advanced(df_columns, signal_role, case_sensitive=False, fuzzy_match=True, substring_match=True)


def find_signal_by_role(mdf, signal_role: str) -> Optional[str]:
    """
    Convenience wrapper for MDF objects.
    
    Args:
        mdf: MDF object from asammdf
        signal_role: Signal role to find (e.g., "rpm", "lambda")
        
    Returns:
        Channel name or None
    """
    if mdf is None or not hasattr(mdf, 'channels_db'):
        return None
    
    available_channels = list(mdf.channels_db.keys())
    return find_signal_advanced(available_channels, signal_role, fuzzy_match=True, substring_match=True)


if __name__ == "__main__":
    # Test the signal mapping
    test_channels = [
        "Epm_nEng_RTE",
        "TqSys_tqCkEngReal",
        "ExM_Lam_Estim",
        "CoolantTemp",
        "Vehicle_Speed",
        "GearPos"
    ]
    
    print("Testing Signal Mapping System")
    print("=" * 60)
    
    stats = get_signal_statistics(test_channels)
    print(f"\nFound Signals: {stats['found_count']}/{stats['total_roles']} ({stats['coverage_percent']}%)")
    print(f"\nFound:")
    for role, channel in stats['found'].items():
        print(f"  {role}: {channel}")
    
    print(f"\nMissing:")
    for role, candidate_count in list(stats['missing'].items())[:5]:
        print(f"  {role}: {candidate_count} candidates checked")

