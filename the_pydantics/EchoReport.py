from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


#
# 1) SUBMODEL FOR ANY NUMERIC FIELD
#

class Z_score_of_reported_numeric(BaseModel):
    z_score: Optional[float] = Field(
        None,
        description="Z-score for the measurement, if available. Represents the number of standard deviations a value is from the mean of a reference population."
    )
    type_of_z_score: Optional[str] = Field(
        None,
        description="Type of z-score used (e.g., 'age-adjusted', 'sex-adjusted', 'population-based', 'Boston', 'PHN', etc.)."
    )


class NumericValue(BaseModel):
    numeric: Optional[float] = Field(
        None,
        description="Numeric measurement value."
    )
    unit: Optional[str] = Field(
        None,
        description="Unit of measurement (e.g., mm, cm, m/s, mmHg, etc.)."
    )
    z_score: Optional[List[Z_score_of_reported_numeric]] = Field(
        None,
        description="Z-score(s) for the measurement, if available."
    )

#
# 2) ENUMS FOR QUALITATIVE DATA
#

class Bool_with_Other(str, Enum):
    TRUE = "True or yes"
    FALSE = "False or No"
    OTHER = "Other"

class DilationSeverity(str, Enum):
    APLASTIC = "Aplastic"
    HYPOPLASTIC = "Hypoplastic"
    SMALL = "Small"
    NORMAL = "Normal"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class HypertrophySeverity(str, Enum):
    NO = "No"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class StenosisSeverity(str, Enum):
    NO = "No"
    TRACE_TRIVIAL = "Trace/Trivial"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class RegurgitationSeverity(str, Enum):
    NO = "No"
    TRACE_TRIVIAL = "Trace/Trivial"
    MILD = "Mild"
    MILD_MODERATE = "Mild-Moderate"
    MODERATE = "Moderate"
    MODERATE_SEVERE = "Moderate-Severe"
    SEVERE = "Severe"

class StructuralStatus(str, Enum):
    NORMAL = "Structurally normal"
    ABNORMAL = "Structurally abnormal"
    OTHER = "Other"

class SystolicDiastolicFunction(str, Enum):
    NORMAL = "Normal"
    LOW_NORMAL = "Low normal"
    MILDLY_DEPRESSED = "Mildly depressed"
    MILD_MODERATE_DEPRESSED = "Mild-Moderately depressed"
    MODERATELY_DEPRESSED = "Moderately depressed"
    MODERATE_SEVERE_DEPRESSED = "Moderate-Severely depressed"
    SEVERELY_DEPRESSED = "Severely depressed"
    AKINETIC = "Akinetic"
    HYPERDYNAMIC = "Hyperdynamic"
    OTHER = "Other"

class PressureSeverity(str, Enum):
    NORMAL = "Normal"
    MILD_ELEVATED = "Mildly elevated"
    MODERATE_ELEVATED = "Moderately elevated"
    SEVERE_ELEVATED = "Severely elevated"
    LESS_HALF_SYSTEMIC = "Less than half systemic"
    HALF_SYSTEMIC = "Half systemic"
    APPROACHING_SYSTEMIC = "Approaching systemic"
    SYSTEMIC = "Systemic"
    SUPRASYSTEMIC = "Suprasystemic"

class Size(str, Enum):
    TINY = "Tiny"
    SMALL = "Small"
    SMALL_MODERATE = "Small-moderate" 
    MODERATE = "Moderate"
    MODERATE_LARGE = "Moderate-large"
    LARGE = "Large"


class ASD_type(str, Enum):
    SECUNDUM = "Secundum"
    PRIMUM = "Primum"
    PFO = "Patent Foramen Ovale"
    SUPERIOR_SINUS_VENOSUS = "Superior sinus venosus"
    CORONARY_SINUS = "Coronary sinus"
    SURGICAL = "Surgically created atrial septal defect"
    INFERIOR_SINUS_VENOSUS = "Inferior sinus venosus"
    OTHER = "Other"

class VSD_type(str, Enum):
    PERIMEMBRANOUS = "Perimembranous"
    ANTERIOR_MALALIGNMENT = "Anterior malalignment"
    POSTERIOR_MALALIGNMENT = "Posterior malalignment"
    MID_MUSCULAR = "Mid muscular"
    POSTERIOR_MUSCULAR = "Posterior muscular"
    APICAL_MUSCULAR = "Apical muscular"
    INLET = "Inlet"
    OUTLET_DOUBLY_COMMITTED = "Outlet/doubly-committed juxta-arterial"
    PERIMEMBRANOUS_INLET = "Perimembranous inlet"
    OUTLET_SUBAORTIC = "Outlet/subaortic"
    CONOVENTRICULAR = "Conoventricular"
    OTHER = "Other"

#
# 3) DEFINE SUBMODELS FOR EACH ANATOMIC REGION
#

# Atria

class RA_info(BaseModel):
    """Right Atrium"""
    RA_dilation: Optional[DilationSeverity] = Field(
        None, 
        description="Right atrial (RA) dilation/size severity."
    )

    


class LA_info(BaseModel):
    """Left Atrium"""
    LA_dilation: Optional[DilationSeverity] = Field(
        None, 
        description="Left atrial (LA) dilation/size severity."
    )
    LA_volume_indexed: Optional[NumericValue] = Field(
        None, 
        description="Left atrial (LA) volume measurement (indexed or unindexed, depending on usage)."
    )

class Atria(BaseModel):
    """Combined submodel for Atria"""
    RA: Optional[RA_info] = None
    LA: Optional[LA_info] = None


# Ventricles

class RVSizeStructure(BaseModel):
    """Right ventricle size/structure details."""
    RV_dilation: Optional[DilationSeverity] = Field(
        None,
        description="Right ventricle (RV) dilation presence/severity."
    )
    RV_hypertrophy: Optional[HypertrophySeverity] = Field(
        None,
        description="Right ventricle (RV) hypertrophy presence/severity."
    )

class RVFunction(BaseModel):
    """Right ventricle functional details."""
    RV_systolic_function: Optional[SystolicDiastolicFunction] = Field(
        None,
        description="Qualitative right ventricular (RV) systolic function."
    )

class RV_info(BaseModel):
    """Combines size/structure and function for the RV."""
    RV_size_structure: Optional[RVSizeStructure] = None
    RV_function: Optional[RVFunction] = None


class LVSizeStructure(BaseModel):
    """Left ventricle size/structure details."""
    LV_dilation: Optional[DilationSeverity] = Field(
        None,
        description="Left ventricle (LV) dilation presence/severity."
    )
    LV_hypertrophy: Optional[HypertrophySeverity] = Field(
        None,
        description="Left ventricle (LV) hypertrophy presence/severity."
    )
    LV_volume_systole: Optional[NumericValue] = Field(
        None,
        description="Left ventricle (LV) volume in systole."
    )
    LV_volume_diastole: Optional[NumericValue] = Field(
        None,
        description="Left ventricle (LV) volume in diastole."
    )

class LVFunction(BaseModel):
    """Left ventricle functional details."""
    LV_systolic_function: Optional[SystolicDiastolicFunction] = Field(
        None,
        description="Qualitative LV systolic function."
    )
    LV_systolic_function_other: Optional[str] = Field(
        None,
        description="If LV_systolic_function is 'Other', specify here."
    )
    LVEF: Optional[NumericValue] = Field(
        None,
        description="Left Ventricular Ejection Fraction (e.g., Simpson's biplane)."
    )

class LV_info(BaseModel):
    """Combines size/structure and function for the LV."""
    LV_size_structure: Optional[LVSizeStructure] = None
    LV_function: Optional[LVFunction] = None


class Ventricles(BaseModel):
    """Container for both right and left ventricles."""
    RV: Optional[RV_info] = None
    LV: Optional[LV_info] = None


# Valves

class TricuspidValve(BaseModel):
    """Tricuspid valve details."""
    TV_structural_status: Optional[StructuralStatus] = Field(
        None,
        description="Structurally normal or abnormal tricuspid valve."
    )
    TV_structural_status_other: Optional[str] = Field(
        None,
        description="If TV_structural_status is 'Other', specify here."
    )
    TV_regurgitation_severity: Optional[RegurgitationSeverity] = Field(
        None,
        description="Tricuspid regurgitation presence/severity."
    )

class PulmonaryValve(BaseModel):
    """Pulmonary valve details."""
    PV_annulus_size: Optional[NumericValue] = Field(
        None,
        description="Pulmonary valve annulus diameter/size."
    )
    PV_stenosis_severity: Optional[StenosisSeverity] = Field(
        None,
        description="Pulmonary valve stenosis presence/severity."
    )
    PV_structural_status: Optional[StructuralStatus] = Field(
        None,
        description="Pulmonary valve structure normal/abnormal."
    )
    PV_structural_status_other: Optional[str] = Field(
        None,
        description="If PV_structural_status is 'Other', specify here."
    )
    PV_regurgitation_severity: Optional[RegurgitationSeverity] = Field(
        None,
        description="Pulmonary valve regurgitation severity."
    )
    PV_pressure_gradient: Optional[NumericValue] = Field(
        None,
        description="Peak pressure gradient across the pulmonary valve."
    )

class MitralValve(BaseModel):
    """Mitral valve details."""
    MV_stenosis_severity: Optional[StenosisSeverity] = Field(
        None,
        description="Mitral valve stenosis presence/severity."
    )
    MV_structural_status: Optional[StructuralStatus] = Field(
        None,
        description="Mitral valve structure normal/abnormal."
    )
    MV_structural_status_other: Optional[str] = Field(
        None,
        description="If MV_structural_status is 'Other', specify here."
    )
    MV_regurgitation_severity: Optional[RegurgitationSeverity] = Field(
        None,
        description="Mitral regurgitation presence/severity."
    )

class AorticValve(BaseModel):
    """Aortic valve details."""
    AV_structural_status: Optional[StructuralStatus] = Field(
        None,
        description="Aortic valve structure normal/abnormal."
    )
    AV_structural_status_other: Optional[str] = Field(
        None,
        description="If AV_structural_status is 'Other', specify here."
    )
    AV_leaflets: Optional[int] = Field(
        None,
        description="Number of leaflets (e.g., 2 or 3)."
    )
    AV_stenosis_severity: Optional[StenosisSeverity] = Field(
        None,
        description="Aortic valve stenosis presence/severity."
    )
    AV_regurgitation_severity: Optional[RegurgitationSeverity] = Field(
        None,
        description="Aortic regurgitation presence/severity."
    )
    AV_peak_pressure_gradient: Optional[NumericValue] = Field(
        None,
        description="Peak measured pressure gradient across the aortic valve."
    )
    AV_mean_pressure_gradient: Optional[NumericValue] = Field(
        None,
        description="Mean measured pressure gradient across the aortic valve."
    )

class Valves(BaseModel):
    """Top-level container for all valves."""
    tricuspid: Optional[TricuspidValve] = None
    pulmonary: Optional[PulmonaryValve] = None
    mitral: Optional[MitralValve] = None
    aortic: Optional[AorticValve] = None


# Great Vessels

class Aorta(BaseModel):
    """Aorta details."""
    arch_sidedness: Optional[str] = Field(
        None,
        description="For example, Left arch or Right arch."
    )
    aortic_root_size: Optional[NumericValue] = Field(
        None,
        description="Aortic root dimension."
    )
    ascending_aorta_diameter: Optional[NumericValue] = Field(
        None,
        description="Ascending aorta diameter."
    )
    aortic_isthmus_size: Optional[NumericValue] = Field(
        None,
        description="Aortic isthmus dimension."
    )
    coarctation: Optional[bool] = Field(
        None,
        description="True if there is any aortic coarctation."
    )
    coarctation_gradient: Optional[NumericValue] = Field(
        None,
        description="Peak pressure gradient across the coarctation, if present."
    )

class GreatVessels(BaseModel):
    """Container for great vessel details."""
    aorta: Optional[Aorta] = None


# Pulmonary Hypertension

class PulmonaryHypertension(BaseModel):
    """Pulmonary hypertension assessment."""
    severity: Optional[str] = Field(
        None,
        description="None, mild, moderate, severe, half-systemic, etc."
    )
    TR_jet_gradient: Optional[NumericValue] = Field(
        None,
        description="Tricuspid regurgitation jet gradient (mmHg)."
    )
    IVS_flattening_in_systole: Optional[bool] = Field(
        None,
        description="True if interventricular septal flattening occurs in systole."
    )


# Atrial communication

class ASD(BaseModel):
    """Atrial communication (ASD/PFO) details currently present."""
    atrial_communication_present: Optional[bool] = Field(
        None,
        description="True if a current ASD is present."
    )
    asd_types: Optional[List[ASD_type]] = Field(
        None,
        description="List of atrial communication types if multiple are present."
    )
    atrial_communication_present_other: Optional[str] = Field(
        None,
        description="If 'Other' was selected in ASD_type, specify here."
    )
    size: Optional[Size] = Field(
        None, 
        description="Size of the atrial communication (e.g., largest one if multiple)."
    )
    direction_of_flow: Optional[str] = Field(
        None, 
        description="Direction of shunting across the atrial communication."
    )


# Ventricular communication

class VSD(BaseModel):
    """Ventricular septal defect (VSD) details currently present."""
    ventricular_communication_present: Optional[bool] = Field(
        None,
        description="True if a current VSD is present."
    )
    vsd_types: Optional[List[VSD_type]] = Field(
        None,
        description="List of VSD types if multiple are present."
    )
    ventricular_communication_present_other: Optional[str] = Field(
        None,
        description="If 'Other' was selected in VSD_type, specify here."
    )
    size: Optional[Size] = Field(
        None, 
        description="Size of the VSD (largest if multiple)."
    )
    direction_of_flow: Optional[str] = Field(
        None, 
        description="Direction of shunting across the VSD."
    )
    peak_gradient: Optional[NumericValue] = Field(
        None,
        description="Peak pressure gradient across the VSD."
    )


# PDA

class PDA(BaseModel):
    """Patent ductus arteriosus details."""
    present: Optional[bool] = Field(
        None,
        description="True if a PDA is currently present."
    )
    direction_of_flow: Optional[str] = Field(
        None,
        description="Direction of shunting across the PDA."
    )
    size: Optional[NumericValue] = Field(
        None, 
        description="Patent ductus arteriosus size."
    )
    peak_gradient: Optional[NumericValue] = Field(
        None,
        description="Peak pressure gradient across the PDA."
    )


# Surgical / Device History

class SurgicalHistory(BaseModel):
    """Surgical or interventional history."""
    prior_surgical_interventions: Optional[str] = Field(
        None, 
        description="Details of previous surgeries or repairs."
    )


#
# 4) TOP-LEVEL ECHO REPORT MODEL
#



class EchoReport(BaseModel):
    """Main Pydantic model capturing the full echo report."""

    atria: Optional[Atria] = None
    ventricles: Optional[Ventricles] = None
    valves: Optional[Valves] = None
    great_vessels: Optional[GreatVessels] = None
    pHTN: Optional[PulmonaryHypertension] = None

    # Shunts / Additional findings
    asd: Optional[ASD] = None
    vsd: Optional[VSD] = None
    pda: Optional[PDA] = None

    # Additional structural/surgical/device data
    surgical_history: Optional[SurgicalHistory] = None

    class Config:
        # If you prefer enum field values as strings in the JSON output:
        use_enum_values = True
        extract_json= {
            "example": {
                "atria": {
                    "RA": {"RA_dilation": "Mild"},
                    "LA": {
                        "LA_dilation": "Moderate",
                        "LA_volume_indexed": {
                            "numeric": 35.0,
                            "unit": "mL/m^2"
                        }
                    }
                },
                "ventricles": {
                    "RV": {
                        "RV_size_structure": {
                            "RV_dilation": "Normal",
                            "RV_hypertrophy": "No"
                        },
                        "RV_function": {
                            "RV_systolic_function": "Normal"
                        }
                    },
                    "LV": {
                        "LV_size_structure": {
                            "LV_dilation": "Mild",
                            "LV_hypertrophy": "Mild",
                            "LV_volume_systole": {
                                "numeric": 50.0,
                                "unit": "mL"
                            },
                            "LV_volume_diastole": {
                                "numeric": 120.0,
                                "unit": "mL"
                            }
                        },
                        "LV_function": {
                            "LV_systolic_function": "Normal",
                            "LVEF": {
                                "numeric": 60.0,
                                "unit": "%"
                            }
                        }
                    }
                },
                "valves": {
                    "tricuspid": {
                        "TV_structural_status": "Structurally normal",
                        "TV_regurgitation_severity": "Mild"
                    },
                    "pulmonary": {
                        "PV_annulus_size": {
                            "numeric": 18.0,
                            "unit": "mm"
                        },
                        "PV_stenosis_severity": "No",
                        "PV_structural_status": "Structurally normal",
                        "PV_regurgitation_severity": "Trace/Trivial"
                    },
                    "mitral": {
                        "MV_stenosis_severity": "No",
                        "MV_structural_status": "Structurally normal",
                        "MV_regurgitation_severity": "Mild"
                    },
                    "aortic": {
                        "AV_structural_status": "Structurally normal",
                        "AV_leaflets": 3,
                        "AV_stenosis_severity": "No",
                        "AV_regurgitation_severity": "No"
                    }
                },
                "great_vessels": {
                    "aorta": {
                        "arch_sidedness": "Left",
                        "aortic_root_size": {
                            "numeric": 28.0,
                            "unit": "mm"
                        },
                        "ascending_aorta_diameter": {
                            "numeric": 30.0,
                            "unit": "mm"
                        },
                        "aortic_isthmus_size": {
                            "numeric": 18.0,
                            "unit": "mm"
                        },
                        "coarctation": False
                    }
                },
                "pHTN": {
                    "severity": "None",
                    "TR_jet_gradient": {
                        "numeric": 20.0,
                        "unit": "mmHg"
                    },
                    "IVS_flattening_in_systole": False
                },
                "asd": {
                    "atrial_communication_present": True,
                    "asd_types": ["PFO", "Secundum"],
                    "size": "Small",
                    "direction_of_flow": "Left to right"
                },
                "vsd": {
                    "ventricular_communication_present": True,
                    "vsd_types": ["PERIMEMBRANOUS", "MID_MUSCULAR"],
                    "size": "Moderate",
                    "direction_of_flow": "Left to right",
                    "peak_gradient": {
                        "numeric": 40,
                        "unit": "mmHg"
                    }
                },
                "pda": {
                    "present": False
                },
                "surgical_history": {
                    "prior_surgical_interventions": "No prior surgeries"
                }
            }
        }