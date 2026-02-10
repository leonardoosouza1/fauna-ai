# Arquitetura Cognitiva IARA v10 — CfC, NCP, GWT, Plasticidade
from src.systems.brain.cfc_cell import CfCCell
from src.systems.brain.ncp import NCPWiring, NCPCell, build_ncp_wiring
from src.systems.brain.policy import AgentBrain, SensoryEncoder
from src.systems.brain.workspace import (
    CrossAttentionFusion,
    ConsensusFusion,
    GlobalWorkspace,
    WorkspaceModulator,
)
from src.systems.brain.plasticity import (
    hebbian_update,
    oja_update,
    stdp_update,
    homeostatic_scaling,
)
