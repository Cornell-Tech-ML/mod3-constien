# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

----

<h1>Parallel Check</h1>

FastOps Parallel Check (Map, Zip, Reduce, and MatMul): to avoid markdown formatting incompatibilities, the output of `python parallel_check.py` is included in the text file `parallel_check.txt`

<h1>Benchmarking</h1>

An average of elapsed time per 10 epochs is located at the beginning of each summary

<h2>Simple</h2>

<b>CPU</b>

avg elapsed time per 10 epochs: 1.740276427268982

elapsed time (s): 22.358 Epoch  0  loss  5.486689395942493 correct 40

elapsed time (s): 1.206 Epoch  10  loss  2.5993732950022066 correct 48

elapsed time (s): 1.190 Epoch  20  loss  1.436065191812198 correct 49

elapsed time (s): 1.210 Epoch  30  loss  1.1516439157550586 correct 49

elapsed time (s): 1.202 Epoch  40  loss  0.5534687979026323 correct 49

elapsed time (s): 1.216 Epoch  50  loss  0.4284245411168604 correct 50

elapsed time (s): 1.546 Epoch  60  loss  0.2590763825875813 correct 50

elapsed time (s): 1.212 Epoch  70  loss  0.256805777229975 correct 50

elapsed time (s): 1.413 Epoch  80  loss  0.40502978412515633 correct 50

elapsed time (s): 2.134 Epoch  90  loss  1.330038233838514 correct 50

elapsed time (s): 1.207 Epoch  100  loss  0.05507674317022072 correct 50

elapsed time (s): 1.212 Epoch  110  loss  0.1354491915027492 correct 50

elapsed time (s): 1.190 Epoch  120  loss  0.3865496997625526 correct 50

elapsed time (s): 1.205 Epoch  130  loss  0.2175826437312148 correct 50

elapsed time (s): 1.222 Epoch  140  loss  0.040411302447803124 correct 50

elapsed time (s): 1.247 Epoch  150  loss  0.08474630659035676 correct 50

elapsed time (s): 1.201 Epoch  160  loss  0.11222906857814266 correct 50

elapsed time (s): 1.199 Epoch  170  loss  0.6064372607023153 correct 50

elapsed time (s): 2.125 Epoch  180  loss  0.6257904537145249 correct 50

elapsed time (s): 1.468 Epoch  190  loss  0.2986510108943287 correct 50

elapsed time (s): 1.194 Epoch  200  loss  0.4018855346103598 correct 50

elapsed time (s): 1.179 Epoch  210  loss  0.06844394584517215 correct 50

elapsed time (s): 1.185 Epoch  220  loss  0.18404082281206868 correct 50

elapsed time (s): 1.187 Epoch  230  loss  0.0283153643725544 correct 50

elapsed time (s): 1.182 Epoch  240  loss  0.15052868159025015 correct 50

elapsed time (s): 1.187 Epoch  250  loss  0.32525529214926324 correct 50

elapsed time (s): 1.220 Epoch  260  loss  0.11842789004051121 correct 50

elapsed time (s): 1.552 Epoch  270  loss  0.11896347536913761 correct 50

elapsed time (s): 1.922 Epoch  280  loss  0.33095047265014643 correct 50

elapsed time (s): 1.185 Epoch  290  loss  0.0278086539374686 correct 50

elapsed time (s): 1.187 Epoch  300  loss  0.009760843075962386 correct 50

elapsed time (s): 1.209 Epoch  310  loss  0.7537166051138606 correct 50

elapsed time (s): 1.192 Epoch  320  loss  0.22833710750185143 correct 50

elapsed time (s): 1.179 Epoch  330  loss  0.052450647151250494 correct 50

elapsed time (s): 1.200 Epoch  340  loss  0.2694882539664689 correct 50

elapsed time (s): 1.183 Epoch  350  loss  0.12164120174547158 correct 50

elapsed time (s): 1.187 Epoch  360  loss  0.04976581579868031 correct 50

elapsed time (s): 1.990 Epoch  370  loss  0.14649458688113765 correct 50

elapsed time (s): 1.589 Epoch  380  loss  0.010485727542573039 correct 50

elapsed time (s): 1.186 Epoch  390  loss  0.0005915133883909181 correct 50

elapsed time (s): 1.177 Epoch  400  loss  0.1666871037067517 correct 50

elapsed time (s): 1.174 Epoch  410  loss  0.04934238546705292 correct 50

elapsed time (s): 1.188 Epoch  420  loss  0.04502760014459002 correct 50

elapsed time (s): 1.188 Epoch  430  loss  0.1472157185093427 correct 50

elapsed time (s): 1.176 Epoch  440  loss  0.20584720575844956 correct 50

elapsed time (s): 1.192 Epoch  450  loss  0.00019405770200431373 correct 50

elapsed time (s): 1.363 Epoch  460  loss  0.17803793858272698 correct 50

elapsed time (s): 2.139 Epoch  470  loss  0.033765425446507694 correct 50

elapsed time (s): 1.179 Epoch  480  loss  0.1537731034127976 correct 50

elapsed time (s): 1.177 Epoch  490  loss  0.04054921480441125 correct 50

<b>GPU</b>

avg elapsed time per 10 epochs: 19.343419132232665

elapsed time (s): 5.232 Epoch  0  loss  5.428607617545115 correct 45

elapsed time (s): 19.874 Epoch  10  loss  0.6785679082034096 correct 48

elapsed time (s): 20.057 Epoch  20  loss  0.5353159088389541 correct 50

elapsed time (s): 19.216 Epoch  30  loss  0.31376635314769363 correct 50

elapsed time (s): 19.849 Epoch  40  loss  1.14422559782388 correct 50

elapsed time (s): 20.676 Epoch  50  loss  0.9656032528771756 correct 50

elapsed time (s): 19.096 Epoch  60  loss  0.7994418203397256 correct 50

elapsed time (s): 19.799 Epoch  70  loss  0.053681258999823234 correct 50

elapsed time (s): 19.595 Epoch  80  loss  0.7268295108710433 correct 50

elapsed time (s): 19.562 Epoch  90  loss  0.258013686980319 correct 50

elapsed time (s): 19.894 Epoch  100  loss  0.2976434122231116 correct 50

elapsed time (s): 19.223 Epoch  110  loss  0.028551102463690498 correct 50

elapsed time (s): 19.890 Epoch  120  loss  0.7559215906403935 correct 50

elapsed time (s): 19.947 Epoch  130  loss  0.204025890709036 correct 50

elapsed time (s): 19.091 Epoch  140  loss  0.47170441654724654 correct 50

elapsed time (s): 19.820 Epoch  150  loss  0.20135159529014987 correct 50

elapsed time (s): 19.639 Epoch  160  loss  0.018297512431720987 correct 50

elapsed time (s): 19.418 Epoch  170  loss  0.47069350541506194 correct 50

elapsed time (s): 20.574 Epoch  180  loss  0.4978484276530707 correct 50

elapsed time (s): 19.226 Epoch  190  loss  0.337981260596904 correct 50

elapsed time (s): 19.634 Epoch  200  loss  0.39225115719206927 correct 50

elapsed time (s): 19.777 Epoch  210  loss  0.4335601396821631 correct 50

elapsed time (s): 19.043 Epoch  220  loss  0.5869437241894859 correct 50

elapsed time (s): 19.863 Epoch  230  loss  0.22334067924359421 correct 50

elapsed time (s): 19.886 Epoch  240  loss  0.5286194325011998 correct 50

elapsed time (s): 19.029 Epoch  250  loss  0.21265054127947025 correct 50

elapsed time (s): 19.791 Epoch  260  loss  0.227715607739726 correct 50

elapsed time (s): 19.315 Epoch  270  loss  0.19087943575750133 correct 50

elapsed time (s): 19.673 Epoch  280  loss  0.35343098572022774 correct 50

elapsed time (s): 19.940 Epoch  290  loss  0.11111414866779283 correct 50

elapsed time (s): 19.157 Epoch  300  loss  0.15892362066523172 correct 50

elapsed time (s): 20.649 Epoch  310  loss  0.184550967397405 correct 50

elapsed time (s): 19.885 Epoch  320  loss  0.0245188115203784 correct 50

elapsed time (s): 19.054 Epoch  330  loss  0.1800443843586953 correct 50

elapsed time (s): 19.755 Epoch  340  loss  0.2536366824720995 correct 50

elapsed time (s): 19.143 Epoch  350  loss  0.024731691257612706 correct 50

elapsed time (s): 19.689 Epoch  360  loss  0.17701733406092554 correct 50

elapsed time (s): 19.761 Epoch  370  loss  0.10892363962752313 correct 50

elapsed time (s): 19.043 Epoch  380  loss  0.17043143871680747 correct 50

elapsed time (s): 19.714 Epoch  390  loss  0.00022243431493296358 correct 50

elapsed time (s): 19.644 Epoch  400  loss  0.041388114060158396 correct 50

elapsed time (s): 19.068 Epoch  410  loss  0.0013855624357417132 correct 50

elapsed time (s): 19.921 Epoch  420  loss  0.0019812600991487876 correct 50

elapsed time (s): 19.092 Epoch  430  loss  0.0049223823440388585 correct 50

elapsed time (s): 20.494 Epoch  440  loss  0.07431869845293607 correct 50

elapsed time (s): 19.924 Epoch  450  loss  0.08271559425530305 correct 50

elapsed time (s): 19.081 Epoch  460  loss  0.1330144927789117 correct 50

elapsed time (s): 19.697 Epoch  470  loss  0.0017197541466442113 correct 50

elapsed time (s): 19.563 Epoch  480  loss  0.17543675974290107 correct 50

elapsed time (s): 19.210 Epoch  490  loss  0.03329655173746427 correct 50

<h2>Split</h2>

<b>CPU</b>

avg elapsed time per 10 epochs: 1.7104537963867188

elapsed time (s): 20.979 Epoch  0  loss  13.620567482262686 correct 14

elapsed time (s): 2.230 Epoch  10  loss  6.0767865288977045 correct 39

elapsed time (s): 1.199 Epoch  20  loss  3.3012933129397495 correct 39

elapsed time (s): 1.194 Epoch  30  loss  4.165984099969711 correct 43

elapsed time (s): 1.193 Epoch  40  loss  3.7859114394158375 correct 47

elapsed time (s): 1.215 Epoch  50  loss  1.7040531856684182 correct 44

elapsed time (s): 1.198 Epoch  60  loss  2.8674462205073814 correct 44

elapsed time (s): 1.187 Epoch  70  loss  1.2233451142406677 correct 46

elapsed time (s): 1.190 Epoch  80  loss  0.6577064625252143 correct 45

elapsed time (s): 1.203 Epoch  90  loss  1.2435485630792318 correct 49

elapsed time (s): 1.561 Epoch  100  loss  1.40180428576794 correct 50

elapsed time (s): 1.744 Epoch  110  loss  2.472997205464135 correct 48

elapsed time (s): 1.183 Epoch  120  loss  2.0042027759533485 correct 50

elapsed time (s): 1.194 Epoch  130  loss  0.6739763468645797 correct 50

elapsed time (s): 1.187 Epoch  140  loss  0.745217220535488 correct 50

elapsed time (s): 1.195 Epoch  150  loss  0.2607025383360442 correct 50

elapsed time (s): 1.198 Epoch  160  loss  1.63501092320111 correct 50

elapsed time (s): 1.207 Epoch  170  loss  1.3081120724295032 correct 49

elapsed time (s): 1.193 Epoch  180  loss  0.2954028606861393 correct 50

elapsed time (s): 1.229 Epoch  190  loss  0.4878428320454651 correct 50

elapsed time (s): 2.234 Epoch  200  loss  0.2130775753597151 correct 50

elapsed time (s): 1.166 Epoch  210  loss  0.27695186506976854 correct 49

elapsed time (s): 1.176 Epoch  220  loss  0.4854153421134595 correct 50

elapsed time (s): 1.170 Epoch  230  loss  0.28044594791508276 correct 50

elapsed time (s): 1.187 Epoch  240  loss  0.4161172497371951 correct 50

elapsed time (s): 1.200 Epoch  250  loss  0.1482123977606515 correct 50

elapsed time (s): 1.186 Epoch  260  loss  0.30670402524394225 correct 50

elapsed time (s): 1.200 Epoch  270  loss  0.18339629776491834 correct 50

elapsed time (s): 1.190 Epoch  280  loss  0.13916682567859104 correct 50

elapsed time (s): 1.984 Epoch  290  loss  0.30821468392654383 correct 50

elapsed time (s): 1.487 Epoch  300  loss  0.14161570599426962 correct 50

elapsed time (s): 1.185 Epoch  310  loss  1.1354886839379426 correct 50

elapsed time (s): 1.209 Epoch  320  loss  0.6119352182404406 correct 50

elapsed time (s): 1.184 Epoch  330  loss  0.5770227142129559 correct 50

elapsed time (s): 1.201 Epoch  340  loss  0.12685834460553633 correct 50

elapsed time (s): 1.191 Epoch  350  loss  0.16042772789864623 correct 50

elapsed time (s): 1.183 Epoch  360  loss  0.18460811451997636 correct 50

elapsed time (s): 1.178 Epoch  370  loss  0.36538925026669544 correct 50

elapsed time (s): 1.557 Epoch  380  loss  0.18068386757997154 correct 50

elapsed time (s): 1.864 Epoch  390  loss  0.3448565247703153 correct 50

elapsed time (s): 1.189 Epoch  400  loss  0.6710208721995923 correct 50

elapsed time (s): 1.179 Epoch  410  loss  0.598227406336984 correct 50

elapsed time (s): 1.181 Epoch  420  loss  0.6980105184728383 correct 50

elapsed time (s): 1.186 Epoch  430  loss  0.6134007169075382 correct 50

elapsed time (s): 1.179 Epoch  440  loss  0.5367255197481073 correct 50

elapsed time (s): 1.202 Epoch  450  loss  0.558402243865649 correct 50

elapsed time (s): 1.187 Epoch  460  loss  0.7857945778395636 correct 50

elapsed time (s): 1.175 Epoch  470  loss  0.2772485863458028 correct 50

elapsed time (s): 2.257 Epoch  480  loss  0.01626144733648569 correct 50

elapsed time (s): 1.177 Epoch  490  loss  0.10575182488687684 correct 50

<b>GPU</b>

avg elapsed time per 10 epochs: 19.336173067092897

elapsed time (s): 4.139 Epoch  0  loss  9.386953448141107 correct 32

elapsed time (s): 20.780 Epoch  10  loss  5.887342026836603 correct 37

elapsed time (s): 19.104 Epoch  20  loss  4.92924067759215 correct 36

elapsed time (s): 19.816 Epoch  30  loss  4.6989851237677875 correct 35

elapsed time (s): 19.747 Epoch  40  loss  3.6063056227311074 correct 44

elapsed time (s): 19.302 Epoch  50  loss  3.313724375356466 correct 46

elapsed time (s): 19.775 Epoch  60  loss  2.93291159508101 correct 44

elapsed time (s): 18.988 Epoch  70  loss  3.441695801331509 correct 43

elapsed time (s): 19.734 Epoch  80  loss  1.9967329909216578 correct 48

elapsed time (s): 19.718 Epoch  90  loss  2.631529923959717 correct 49

elapsed time (s): 19.568 Epoch  100  loss  2.5483660592961663 correct 49

elapsed time (s): 19.766 Epoch  110  loss  2.429309984614828 correct 48

elapsed time (s): 19.558 Epoch  120  loss  2.7581065851983233 correct 47

elapsed time (s): 19.389 Epoch  130  loss  2.27715957579532 correct 48

elapsed time (s): 20.629 Epoch  140  loss  1.849051847459882 correct 49

elapsed time (s): 19.202 Epoch  150  loss  1.9629392397517877 correct 48

elapsed time (s): 19.928 Epoch  160  loss  1.222654070873504 correct 49

elapsed time (s): 19.794 Epoch  170  loss  0.8197961130272844 correct 49

elapsed time (s): 18.918 Epoch  180  loss  0.9641121688973613 correct 49

elapsed time (s): 19.771 Epoch  190  loss  3.0524304509153 correct 46

elapsed time (s): 19.287 Epoch  200  loss  0.5784959504178029 correct 49

elapsed time (s): 19.471 Epoch  210  loss  1.1947337278250902 correct 48

elapsed time (s): 19.784 Epoch  220  loss  0.6467830819962787 correct 48

elapsed time (s): 19.027 Epoch  230  loss  0.7465407169589685 correct 50

elapsed time (s): 19.713 Epoch  240  loss  1.4484122892425888 correct 49

elapsed time (s): 19.566 Epoch  250  loss  1.2319258483184707 correct 46

elapsed time (s): 19.677 Epoch  260  loss  0.6178721576961412 correct 50

elapsed time (s): 19.922 Epoch  270  loss  0.5973268090858774 correct 46

elapsed time (s): 19.208 Epoch  280  loss  1.4964258863689297 correct 46

elapsed time (s): 19.554 Epoch  290  loss  0.7156462390818988 correct 48

elapsed time (s): 19.786 Epoch  300  loss  1.2048226562652748 correct 50

elapsed time (s): 19.104 Epoch  310  loss  2.724575781719455 correct 45

elapsed time (s): 19.900 Epoch  320  loss  1.1311941830142438 correct 50

elapsed time (s): 19.798 Epoch  330  loss  1.3129743212251577 correct 50

elapsed time (s): 19.105 Epoch  340  loss  0.6456115569753413 correct 50

elapsed time (s): 19.847 Epoch  350  loss  0.7377670614192289 correct 49

elapsed time (s): 19.210 Epoch  360  loss  1.5114957152746225 correct 48

elapsed time (s): 19.550 Epoch  370  loss  1.2041961741833864 correct 48

elapsed time (s): 19.995 Epoch  380  loss  0.5116943703501838 correct 50

elapsed time (s): 20.060 Epoch  390  loss  1.146699734093502 correct 50

elapsed time (s): 19.931 Epoch  400  loss  1.974465387069197 correct 48

elapsed time (s): 19.971 Epoch  410  loss  1.0944272751151924 correct 49

elapsed time (s): 19.276 Epoch  420  loss  1.0653856683660239 correct 50

elapsed time (s): 20.012 Epoch  430  loss  0.8236968624202393 correct 50

elapsed time (s): 19.889 Epoch  440  loss  1.393609173522377 correct 47

elapsed time (s): 19.319 Epoch  450  loss  0.5377569064482002 correct 50

elapsed time (s): 19.977 Epoch  460  loss  0.1265605150624716 correct 49

elapsed time (s): 19.514 Epoch  470  loss  0.5081025283737012 correct 50

elapsed time (s): 19.761 Epoch  480  loss  0.780190868570825 correct 49

elapsed time (s): 19.969 Epoch  490  loss  0.3101518281202187 correct 49

<h2>Xor</h2>

<b>CPU</b>

avg elapsed time per 10 epochs: 1.7427206754684448

elapsed time (s): 22.462 Epoch  0  loss  9.058502731259091 correct 20

elapsed time (s): 1.224 Epoch  10  loss  4.290738327019399 correct 43

elapsed time (s): 1.216 Epoch  20  loss  3.161178177960781 correct 44

elapsed time (s): 1.542 Epoch  30  loss  3.001859523811666 correct 43

elapsed time (s): 1.965 Epoch  40  loss  3.5590473486832175 correct 45

elapsed time (s): 1.199 Epoch  50  loss  3.0035980913850198 correct 45

elapsed time (s): 1.202 Epoch  60  loss  2.176409313107896 correct 47

elapsed time (s): 1.203 Epoch  70  loss  2.417233771873724 correct 47

elapsed time (s): 1.213 Epoch  80  loss  2.7029978386713003 correct 48

elapsed time (s): 1.220 Epoch  90  loss  1.1573190650224914 correct 47

elapsed time (s): 1.206 Epoch  100  loss  2.2538621413201625 correct 48

elapsed time (s): 1.215 Epoch  110  loss  1.0276951478900909 correct 49

elapsed time (s): 1.299 Epoch  120  loss  2.1150662643976386 correct 49

elapsed time (s): 2.174 Epoch  130  loss  1.9248241513337805 correct 49

elapsed time (s): 1.211 Epoch  140  loss  1.5294288087059216 correct 49

elapsed time (s): 1.201 Epoch  150  loss  1.8971830037115611 correct 49

elapsed time (s): 1.221 Epoch  160  loss  1.0791671917246584 correct 49

elapsed time (s): 1.212 Epoch  170  loss  1.1788360310116694 correct 50

elapsed time (s): 1.224 Epoch  180  loss  1.113570458391238 correct 50

elapsed time (s): 1.203 Epoch  190  loss  1.573382478227286 correct 50

elapsed time (s): 1.197 Epoch  200  loss  1.8110232370916226 correct 49

elapsed time (s): 1.205 Epoch  210  loss  0.7586738191511389 correct 50

elapsed time (s): 2.048 Epoch  220  loss  0.9247795010647683 correct 50

elapsed time (s): 1.439 Epoch  230  loss  0.9797912455279595 correct 50

elapsed time (s): 1.203 Epoch  240  loss  0.25931501402720675 correct 50

elapsed time (s): 1.205 Epoch  250  loss  0.8009896845744601 correct 50

elapsed time (s): 1.203 Epoch  260  loss  0.5072491207498852 correct 50

elapsed time (s): 1.198 Epoch  270  loss  0.5394982222218268 correct 50

elapsed time (s): 1.191 Epoch  280  loss  0.3737921659489186 correct 49

elapsed time (s): 1.194 Epoch  290  loss  0.8048199901438045 correct 50

elapsed time (s): 1.213 Epoch  300  loss  0.6018905120620524 correct 50

elapsed time (s): 1.776 Epoch  310  loss  0.607498027203695 correct 49

elapsed time (s): 1.712 Epoch  320  loss  0.4530241734219179 correct 50

elapsed time (s): 1.204 Epoch  330  loss  0.8277844655534958 correct 49

elapsed time (s): 1.214 Epoch  340  loss  0.46385509722358176 correct 50

elapsed time (s): 1.209 Epoch  350  loss  0.30853760955452264 correct 50

elapsed time (s): 1.212 Epoch  360  loss  0.3880774084525188 correct 50

elapsed time (s): 1.211 Epoch  370  loss  0.4205382751146238 correct 50

elapsed time (s): 1.214 Epoch  380  loss  0.6587396293306576 correct 50

elapsed time (s): 1.200 Epoch  390  loss  0.16091461161716258 correct 50

elapsed time (s): 1.512 Epoch  400  loss  0.43247746700919765 correct 50

elapsed time (s): 2.041 Epoch  410  loss  0.2668965657946394 correct 50

elapsed time (s): 1.217 Epoch  420  loss  0.17361835528318104 correct 50

elapsed time (s): 1.208 Epoch  430  loss  0.1467000368233387 correct 50

elapsed time (s): 1.209 Epoch  440  loss  0.5094171731822769 correct 50

elapsed time (s): 1.216 Epoch  450  loss  0.4171767983057536 correct 50

elapsed time (s): 1.204 Epoch  460  loss  0.1948215008782504 correct 50

elapsed time (s): 1.220 Epoch  470  loss  0.37757896699747673 correct 50

elapsed time (s): 1.203 Epoch  480  loss  0.5169245950372338 correct 50

elapsed time (s): 1.246 Epoch  490  loss  0.08261865748655725 correct 50

<b>GPU</b>

avg elapsed time per 10 epochs: 19.35347464084625

elapsed time (s): 4.990 Epoch  0  loss  6.801109375820451 correct 31

elapsed time (s): 19.239 Epoch  10  loss  5.219749857940737 correct 40

elapsed time (s): 20.087 Epoch  20  loss  3.735443724343523 correct 42

elapsed time (s): 19.513 Epoch  30  loss  4.536711916427517 correct 42

elapsed time (s): 19.537 Epoch  40  loss  4.006579470651317 correct 40

elapsed time (s): 20.015 Epoch  50  loss  3.119723821670423 correct 42

elapsed time (s): 19.271 Epoch  60  loss  3.7977628830592955 correct 45

elapsed time (s): 19.925 Epoch  70  loss  2.173510214261376 correct 43

elapsed time (s): 20.710 Epoch  80  loss  1.8906550403350488 correct 44

elapsed time (s): 19.196 Epoch  90  loss  4.003397741763786 correct 44

elapsed time (s): 19.943 Epoch  100  loss  4.5244511344162825 correct 45

elapsed time (s): 19.964 Epoch  110  loss  3.7975872979962992 correct 44

elapsed time (s): 19.170 Epoch  120  loss  3.8709635784216143 correct 42

elapsed time (s): 19.857 Epoch  130  loss  2.131439339760063 correct 45

elapsed time (s): 19.499 Epoch  140  loss  1.1649036855908388 correct 44

elapsed time (s): 19.617 Epoch  150  loss  2.526020359600526 correct 45

elapsed time (s): 19.871 Epoch  160  loss  2.2145118607447 correct 46

elapsed time (s): 19.138 Epoch  170  loss  2.245226488642662 correct 47

elapsed time (s): 19.913 Epoch  180  loss  1.6831981385326038 correct 47

elapsed time (s): 19.892 Epoch  190  loss  1.1317766379775716 correct 48

elapsed time (s): 19.893 Epoch  200  loss  2.1837715692220674 correct 46

elapsed time (s): 19.834 Epoch  210  loss  2.861225440606839 correct 48

elapsed time (s): 19.673 Epoch  220  loss  2.176198393842117 correct 48

elapsed time (s): 19.285 Epoch  230  loss  0.8526943063675404 correct 48

elapsed time (s): 19.915 Epoch  240  loss  1.4789725915008547 correct 48

elapsed time (s): 19.017 Epoch  250  loss  3.9142821175283906 correct 45

elapsed time (s): 19.963 Epoch  260  loss  3.075919127366956 correct 49

elapsed time (s): 19.976 Epoch  270  loss  0.9066706709281679 correct 48

elapsed time (s): 19.117 Epoch  280  loss  1.5140957638483725 correct 48

elapsed time (s): 19.720 Epoch  290  loss  0.8071757781147385 correct 48

elapsed time (s): 19.764 Epoch  300  loss  1.5695824941678707 correct 47

elapsed time (s): 19.334 Epoch  310  loss  0.9406325738082405 correct 48

elapsed time (s): 19.825 Epoch  320  loss  1.0911099000135938 correct 49

elapsed time (s): 19.886 Epoch  330  loss  5.005193683584071 correct 43

elapsed time (s): 19.846 Epoch  340  loss  0.5210089699566629 correct 47

elapsed time (s): 19.776 Epoch  350  loss  0.8070614023852665 correct 48

elapsed time (s): 19.049 Epoch  360  loss  1.9216340509476255 correct 48

elapsed time (s): 19.749 Epoch  370  loss  2.7396454708892093 correct 50

elapsed time (s): 19.125 Epoch  380  loss  1.332697811713636 correct 50

elapsed time (s): 19.657 Epoch  390  loss  0.6965448737159875 correct 48

elapsed time (s): 19.848 Epoch  400  loss  0.5811890988962957 correct 48

elapsed time (s): 19.045 Epoch  410  loss  1.5547091138406033 correct 48

elapsed time (s): 19.953 Epoch  420  loss  1.4390918512104507 correct 49

elapsed time (s): 19.820 Epoch  430  loss  0.40892261341203323 correct 49

elapsed time (s): 19.068 Epoch  440  loss  1.4483854416844641 correct 50

elapsed time (s): 19.868 Epoch  450  loss  0.9607947840037163 correct 50

elapsed time (s): 20.245 Epoch  460  loss  0.3362810294497451 correct 48

elapsed time (s): 19.374 Epoch  470  loss  0.30355344953522356 correct 47

elapsed time (s): 19.679 Epoch  480  loss  1.0333097843194656 correct 50

elapsed time (s): 19.029 Epoch  490  loss  1.6367567001677583 correct 49

<h1>Increased Hidden</h1>

<h2>Xor w HIDDEN=250</h2>

<b>CPU</b>

avg elapsed time per 10 epochs: 4.346908597946167

elapsed time (s): 22.700 Epoch  0  loss  5.231576854958752 correct 32

elapsed time (s): 3.703 Epoch  10  loss  3.678626150936052 correct 44

elapsed time (s): 3.673 Epoch  20  loss  3.2348506191480233 correct 46

elapsed time (s): 4.712 Epoch  30  loss  0.8157692626901315 correct 47

elapsed time (s): 3.700 Epoch  40  loss  2.3477980914877685 correct 46

elapsed time (s): 3.643 Epoch  50  loss  1.5620056528952306 correct 48

elapsed time (s): 4.712 Epoch  60  loss  1.7862697305415733 correct 47

elapsed time (s): 3.694 Epoch  70  loss  1.1530725827436075 correct 48

elapsed time (s): 3.675 Epoch  80  loss  1.6274473205880522 correct 49

elapsed time (s): 4.716 Epoch  90  loss  1.586026133627032 correct 49

elapsed time (s): 3.629 Epoch  100  loss  1.5086237759055061 correct 49

elapsed time (s): 3.687 Epoch  110  loss  1.2973670820566316 correct 49

elapsed time (s): 4.704 Epoch  120  loss  0.8951157338883216 correct 49

elapsed time (s): 3.683 Epoch  130  loss  0.5856825304149674 correct 49

elapsed time (s): 3.648 Epoch  140  loss  1.9845028381976668 correct 49

elapsed time (s): 4.676 Epoch  150  loss  0.5809164535529351 correct 48

elapsed time (s): 3.583 Epoch  160  loss  1.9068608626698682 correct 49

elapsed time (s): 3.600 Epoch  170  loss  0.7379527599844506 correct 48

elapsed time (s): 4.617 Epoch  180  loss  0.6156700633087457 correct 50

elapsed time (s): 3.625 Epoch  190  loss  1.4347409137550273 correct 48

elapsed time (s): 3.569 Epoch  200  loss  0.22757806707374192 correct 50

elapsed time (s): 4.602 Epoch  210  loss  1.583349308363535 correct 50

elapsed time (s): 3.585 Epoch  220  loss  0.7297852609362029 correct 49

elapsed time (s): 3.566 Epoch  230  loss  0.15791224155461817 correct 49

elapsed time (s): 4.568 Epoch  240  loss  0.02548698120979362 correct 47

elapsed time (s): 3.593 Epoch  250  loss  0.28563191324238557 correct 50

elapsed time (s): 3.566 Epoch  260  loss  1.1534776084775364 correct 50

elapsed time (s): 4.638 Epoch  270  loss  0.39043377311642324 correct 48

elapsed time (s): 3.608 Epoch  280  loss  0.6266549195543614 correct 50

elapsed time (s): 3.661 Epoch  290  loss  0.6504665266727144 correct 50

elapsed time (s): 4.642 Epoch  300  loss  0.0692389709331907 correct 50

elapsed time (s): 4.661 Epoch  310  loss  0.3340246555136798 correct 49

elapsed time (s): 3.586 Epoch  320  loss  0.049217223739797616 correct 50

elapsed time (s): 4.591 Epoch  330  loss  0.9034795206210242 correct 49

elapsed time (s): 3.599 Epoch  340  loss  0.8581021332707699 correct 50

elapsed time (s): 3.573 Epoch  350  loss  0.7782523553049022 correct 50

elapsed time (s): 4.609 Epoch  360  loss  0.7676664475760729 correct 50

elapsed time (s): 3.587 Epoch  370  loss  0.1501172056683397 correct 50

elapsed time (s): 3.595 Epoch  380  loss  0.48042782527064143 correct 49

elapsed time (s): 4.612 Epoch  390  loss  1.0727113487848856 correct 50

elapsed time (s): 3.600 Epoch  400  loss  1.274020182549609 correct 50

elapsed time (s): 3.580 Epoch  410  loss  0.982881193345841 correct 50

elapsed time (s): 4.593 Epoch  420  loss  0.538372396168135 correct 50

elapsed time (s): 3.644 Epoch  430  loss  0.13044037401429848 correct 48

elapsed time (s): 3.604 Epoch  440  loss  0.9859605225216003 correct 50

elapsed time (s): 4.255 Epoch  450  loss  0.13307214265397715 correct 49

elapsed time (s): 3.926 Epoch  460  loss  1.0127869612901907 correct 50

elapsed time (s): 3.545 Epoch  470  loss  0.8539995574684666 correct 50

elapsed time (s): 3.873 Epoch  480  loss  0.2821404697071059 correct 50

elapsed time (s): 4.334 Epoch  490  loss  0.6678915601213736 correct 50

<b>GPU</b>

avg elapsed time per 10 epochs: 20.29002738952637

elapsed time (s): 4.444 Epoch  0  loss  32.82965713509777 correct 29

elapsed time (s): 20.713 Epoch  10  loss  2.1729837739541322 correct 43

elapsed time (s): 20.859 Epoch  20  loss  3.6954365112490537 correct 45

elapsed time (s): 20.760 Epoch  30  loss  2.705892927406171 correct 46

elapsed time (s): 20.129 Epoch  40  loss  2.345627888592826 correct 43

elapsed time (s): 20.848 Epoch  50  loss  2.1192108573623516 correct 48

elapsed time (s): 20.795 Epoch  60  loss  4.018030729649749 correct 50

elapsed time (s): 20.260 Epoch  70  loss  0.913780966113521 correct 49

elapsed time (s): 21.523 Epoch  80  loss  2.176969223429423 correct 48

elapsed time (s): 20.835 Epoch  90  loss  0.5280550958638179 correct 49

elapsed time (s): 20.837 Epoch  100  loss  1.4867375558236375 correct 47

elapsed time (s): 20.165 Epoch  110  loss  1.239879637449811 correct 45

elapsed time (s): 20.851 Epoch  120  loss  1.3362250672015232 correct 49

elapsed time (s): 20.835 Epoch  130  loss  0.8877615148021184 correct 49

elapsed time (s): 20.039 Epoch  140  loss  0.7270917011285083 correct 49

elapsed time (s): 20.760 Epoch  150  loss  1.2735582729764305 correct 49

elapsed time (s): 20.848 Epoch  160  loss  0.5029715913754297 correct 49

elapsed time (s): 20.775 Epoch  170  loss  0.5804598889264458 correct 49

elapsed time (s): 20.001 Epoch  180  loss  0.4221037503258646 correct 49

elapsed time (s): 20.748 Epoch  190  loss  0.328537056112834 correct 50

elapsed time (s): 21.546 Epoch  200  loss  1.6889872312510532 correct 50

elapsed time (s): 20.217 Epoch  210  loss  0.5219498328788086 correct 49

elapsed time (s): 20.498 Epoch  220  loss  0.3478428267189054 correct 49

elapsed time (s): 20.857 Epoch  230  loss  0.15947677643460964 correct 49

elapsed time (s): 20.800 Epoch  240  loss  0.2780734222331921 correct 49

elapsed time (s): 19.997 Epoch  250  loss  1.2624985517109981 correct 50

elapsed time (s): 20.844 Epoch  260  loss  0.171154461688845 correct 50

elapsed time (s): 20.726 Epoch  270  loss  1.9274804766176403 correct 50

elapsed time (s): 19.970 Epoch  280  loss  0.2749921091093702 correct 50

elapsed time (s): 20.745 Epoch  290  loss  0.38019711412609597 correct 49

elapsed time (s): 20.737 Epoch  300  loss  0.3147380092179218 correct 50

elapsed time (s): 20.733 Epoch  310  loss  0.07180681004282842 correct 49

elapsed time (s): 20.835 Epoch  320  loss  0.2088191543516172 correct 49

elapsed time (s): 20.667 Epoch  330  loss  0.27175788108600546 correct 49

elapsed time (s): 20.691 Epoch  340  loss  0.23507546257172907 correct 50

elapsed time (s): 19.957 Epoch  350  loss  1.3548142802314165 correct 50

elapsed time (s): 20.665 Epoch  360  loss  1.1711203038829767 correct 50

elapsed time (s): 20.649 Epoch  370  loss  0.5177110269958995 correct 50

elapsed time (s): 20.584 Epoch  380  loss  0.41455484604831355 correct 50

elapsed time (s): 20.037 Epoch  390  loss  0.060760684552584585 correct 49

elapsed time (s): 20.738 Epoch  400  loss  0.20545295158556778 correct 49

elapsed time (s): 20.781 Epoch  410  loss  0.0882841087455157 correct 50

elapsed time (s): 20.024 Epoch  420  loss  0.17437459418473955 correct 50

elapsed time (s): 20.684 Epoch  430  loss  0.3224142602147713 correct 50

elapsed time (s): 20.608 Epoch  440  loss  0.1506550441698603 correct 50

elapsed time (s): 21.203 Epoch  450  loss  0.09900422593079608 correct 50

elapsed time (s): 20.311 Epoch  460  loss  0.7203621176145977 correct 50

elapsed time (s): 20.670 Epoch  470  loss  0.15459111088766525 correct 50

elapsed time (s): 20.685 Epoch  480  loss  0.04406930196635438 correct 50

elapsed time (s): 20.017 Epoch  490  loss  0.07057161055661171 correct 50
