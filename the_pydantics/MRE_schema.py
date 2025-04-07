from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum



class FirstImpression(str, Enum):
    CROHNS_DISEASE = "CD"
    ULCERATIVE_COLITIS = "UC"
    INFLAMMATORY_BOWEL_DISEASE = "IBD"
    NORMAL = "Normal"
    ENTERITIS = "Enteritis"
    NEOPLASM = "Neoplasm"
    OTHER = "Other abnormalities"


class DiseaseActivity(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive (quiescent)"
    NOT_MENTIONED = "Not mentioned"
    

class MREnterographyReport(BaseModel):

    clinical_information: str = Field(
        ...,
        description="The main impression for doing MR Enterography, e.g., Rule out IBD, abdominal pain, follow up of known IBD, anemia, dyspepsia, etc."
    )
    
    first_impression: FirstImpression = Field(
        ...,
        description="The first impression of the MR Enterography finding."
    )
    
    neoplasm_present: bool = Field(
        ...,
        description="Indicates if there's evidence of neoplasia or malignancy"
    )
    
    neoplasm_type: Optional[str] = Field(
        None,
        description="Type of malignancy, if applicable"
    )

    endometriosis_present: bool = Field(
        ...,
        description="Indicates if there's evidence of endometriosis"
    )
        
    abnormal_intestine_regions: Optional[str] = Field(
        None,
        description="The affected regions of intestine for IBD, CD, and UC, e.g., ileum, left side of colon, left side of colon and terminal ileum."
    )
    
    disease_activity: DiseaseActivity = Field(
        ...,
        description="The activity stage of the disease (IBD, UC, or CD)."
    )
    
    fistula_present: int = Field(
        ...,
        description="Presence of fistula"
    )

    fistula_location: Optional[str] = Field(
        ...,
        description="Location of fistula, if present"
    )
        
    stricture_present: bool = Field(
        ...,
        description="Presence of strictures"
    )

    stricture_location: Optional[str] = Field(
        ...,
        description="Location of strictures, if present"
    )
        
    abscess_present: bool = Field(
        ...,
        description="Presence of abscess"
    )

    abscess_location: Optional[str] = Field(
        ...,
        description="Location of abscess, if present"
    )
        
    previous_surgery_performed: bool = Field(
        ...,
        description="Indicates if there's evidence of previous surgery"
    )
    
    previous_surgery_type: Optional[str] = Field(
        None,
        description="Type of previous surgery, if applicable"
    )
    
    previous_imaging_exists: bool = Field(
        ...,
        description="Indicates the existence of a previous radiologic image and exam"
    )
    
    previous_imaging_type: Optional[str] = Field(
        None,
        description="Type of previous imaging, if applicable"
    )

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "clinical_information": "Rule out IBD",
                "first_impression": "CD",
                "neoplasm_present": False,
                "neoplasm_type": None,
                "endometriosis_present": False,
                "abnormal_intestine_regions": "terminal ileum",
                "disease_activity": "Active",
                "fistula_present": False,
                "fistula_location": "ileo-ileal fistula",
                "stricture_present": True,
                "stricture_location": "terminal ileum within 10cm of ileocecal valve",
                "abscess_present": False,
                "abscess_location": "left posterolateral wall of ileoanal anastomotic junction just above the level of anal canal",
                "previous_surgery_performed": False,
                "previous_surgery_type": None,
                "previous_imaging_exists": True,
                "previous_imaging_type": "MRI"
            }
        }
