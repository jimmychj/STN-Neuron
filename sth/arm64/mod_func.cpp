#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _AXNODE75_reg(void);
extern void _CaT_reg(void);
extern void _Cacum_reg(void);
extern void _HVA_reg(void);
extern void _Ih_reg(void);
extern void _KDR_reg(void);
extern void _Kv31_reg(void);
extern void _Na_reg(void);
extern void _NaL_reg(void);
extern void _PARAK75_reg(void);
extern void _STh_reg(void);
extern void _myions_reg(void);
extern void _sKCa_reg(void);
extern void _xtra_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"AXNODE75.mod\"");
    fprintf(stderr, " \"CaT.mod\"");
    fprintf(stderr, " \"Cacum.mod\"");
    fprintf(stderr, " \"HVA.mod\"");
    fprintf(stderr, " \"Ih.mod\"");
    fprintf(stderr, " \"KDR.mod\"");
    fprintf(stderr, " \"Kv31.mod\"");
    fprintf(stderr, " \"Na.mod\"");
    fprintf(stderr, " \"NaL.mod\"");
    fprintf(stderr, " \"PARAK75.mod\"");
    fprintf(stderr, " \"STh.mod\"");
    fprintf(stderr, " \"myions.mod\"");
    fprintf(stderr, " \"sKCa.mod\"");
    fprintf(stderr, " \"xtra.mod\"");
    fprintf(stderr, "\n");
  }
  _AXNODE75_reg();
  _CaT_reg();
  _Cacum_reg();
  _HVA_reg();
  _Ih_reg();
  _KDR_reg();
  _Kv31_reg();
  _Na_reg();
  _NaL_reg();
  _PARAK75_reg();
  _STh_reg();
  _myions_reg();
  _sKCa_reg();
  _xtra_reg();
}

#if defined(__cplusplus)
}
#endif
