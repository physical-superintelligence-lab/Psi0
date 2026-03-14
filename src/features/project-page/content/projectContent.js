/**
 * Primary content contract for the project page.
 *
 * To adapt this site for another project, replace the values in this file and
 * add the referenced assets under `public/`. The section components consume
 * this object directly and should not need edits for normal project swaps.
 */

const baseUrl = import.meta.env?.BASE_URL ?? "/";
const normalizedBaseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;

const withBase = (path) => {
  if (!path.startsWith("/")) return path;
  return `${normalizedBaseUrl}${path.slice(1)}`;
};

/**
 * @typedef {{
 *   label: string;
 *   href: string;
 * }} Link
 *
 * @typedef {{
 *   label: string;
 *   href: string;
 *   external?: boolean;
 * }} EyebrowLink
 *
 * @typedef {{
 *   title: string;
 *   body: string;
 * }} SectionCopy
 *
 * @typedef {{
 *   step: string;
 *   title: string;
 *   body: string;
 * }} MethodStep
 *
 * @typedef {{
 *   src: string;
 *   alt: string;
 *   title?: string;
 *   kicker?: string;
 *   caption?: string;
 *   body?: string;
 * }} FigureBlock
 *
 * @typedef {{
 *   label: string;
 *   title: string;
 *   src: string;
 *   emphasis?: "feature";
 *   autoplay?: boolean;
 *   summary: string;
 * }} Demo
 *
 * @typedef {{
 *   slug: string;
 *   title: string;
 *   intro: string;
 *   demos: Demo[];
 * }} DemoGroup
 *
 * @typedef {{
 *   key: string;
 *   label: string;
 *   emphasize?: boolean;
 * }} TableColumn
 *
 * @typedef {{
 *   kicker: string;
 *   title: string;
 *   body: string;
 *   figureCaption?: string;
 * }} ExperimentsCopy
 *
 * @typedef {{
 *   eyebrow: EyebrowLink;
 *   title: string;
 *   authors: string[];
 *   authorNote: string;
 *   affiliations: string[];
 *   teaserVideoSrc: string;
 *   teaserVideoAlt: string;
 *   actions: Link[];
 * }} HeroContent
 *
 * @typedef {{
 *   hero: HeroContent;
 *   abstract: SectionCopy & { kicker: string };
 *   demos: {
 *     sectionTitle: string;
 *     groups: DemoGroup[];
 *   };
 *   method: {
 *     sectionTitle: string;
 *     lead: {
 *       kicker: string;
 *       title: string;
 *       body: string;
 *     };
 *     architecture: FigureBlock & {
 *       kicker: string;
 *       title: string;
 *       body?: string;
 *     };
 *     training: {
 *       kicker: string;
 *       title: string;
 *       body: string;
 *     };
 *     steps: MethodStep[];
 *     supportBlocks: FigureBlock[];
 *   };
 *   experiments: {
 *     sectionTitle: string;
 *     benchmark: ExperimentsCopy & {
 *       figure: FigureBlock;
 *       columns: TableColumn[];
 *       rows: Record<string, string>[];
 *     };
 *     ablation: ExperimentsCopy & {
 *       columns: TableColumn[];
 *       rows: Record<string, string>[];
 *     };
 *   };
 *   footer: {
 *     sourceCodeLabel: string;
 *   };
 * }} ProjectPageContent
 */

/** @type {ProjectPageContent} */
export const projectPageContent = {
  hero: {
    eyebrow: {
      label: "PSI Lab",
      href: "https://psi-lab.ai/",
      external: true,
    },
    title:
      "Ψ₀: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation",
    authors: [
      '<a href="https://songlin.github.io/" target="_blank" rel="noreferrer">Songlin Wei</a><sup>1*</sup>',
      '<a href="https://hongyijing.me/" target="_blank" rel="noreferrer">Hongyi Jing</a><sup>1*</sup>',
      '<a href="https://boqian-li.github.io/" target="_blank" rel="noreferrer">Boqian Li</a><sup>1*</sup>',
      '<a href="https://zhenyuzhao.com/" target="_blank" rel="noreferrer">Zhenyu Zhao</a><sup>1*</sup>',
      '<a href="https://pointscoder.github.io/" target="_blank" rel="noreferrer">Jiageng Mao</a><sup>1</sup>',
      '<a href="https://nizhenhao-3.github.io/" target="_blank" rel="noreferrer">Zhenhao Ni</a><sup>1</sup>',
      '<a href="https://hesicheng.net/" target="_blank" rel="noreferrer">Sicheng He</a><sup>1</sup>',
      '<a href="https://jie0530.github.io/" target="_blank" rel="noreferrer">Jie Liu</a><sup>1</sup>',
      '<a href="https://www.xiaweiliu.com/" target="_blank" rel="noreferrer">Xiawei Liu</a><sup>1</sup>',
      "Kaidi Kang<sup>1</sup>",
      "Sheng Zang<sup>1</sup>",
      '<a href="https://weiduoyuan.com/" target="_blank" rel="noreferrer">Weiduo Yuan</a><sup>1</sup>',
      '<a href="https://profiles.stanford.edu/marco-pavone" target="_blank" rel="noreferrer">Marco Pavone</a><sup>2</sup>',
      "Di Huang<sup>3</sup>",
      '<a href="https://yuewang.xyz/" target="_blank" rel="noreferrer">Yue Wang</a><sup>1†</sup>',
    ],
    authorNote:
      "<sup>*</sup> Equal Contribution &nbsp;&nbsp; <sup>†</sup> Corresponding Author",
    affiliations: [
      "1. USC Physical Superintelligence (PSI) Lab",
      "2. NVIDIA",
      "3. WorldEngine",
    ],
    teaserVideoSrc: withBase("/media/psi-0/psi_0_1080p.mp4"),
    teaserVideoAlt: "Ψ₀ teaser video",
    actions: [
      { label: "Watch Demos", href: "#demos" },
      { label: "Paper", href: "https://arxiv.org/abs/2603.12263" },
      { label: "Model", href: "https://huggingface.co/usc-psi-lab/psi-model" },
      {
        label: "Data",
        href: "https://huggingface.co/datasets/usc-psi-lab/psi-data",
      },
      {
        label: "Code",
        href: "https://github.com/physical-superintelligence-lab/Psi0",
      },
    ],
  },
  abstract: {
    kicker: "Abstract",
    title: "",
    body: "We introduce Ψ₀ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks. While existing approaches often attempt to address this fundamental problem by co-training on large and diverse human and humanoid data, we argue that this strategy is suboptimal due to the fundamental kinematic and motion disparities between humans and humanoid robots. Therefore, data efficiency and model performance remain unsatisfactory despite the considerable data volume. To address this challenge, Ψ₀ decouples the learning process to maximize the utility of heterogeneous data sources. Specifically, we propose a staged training paradigm with different learning objectives: first, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations; then, we post-train a flow-based action expert on high-quality humanoid robot data to learn precise robot joint control. Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance. Extensive real-world experiments demonstrate that Ψ₀ achieves the best performance using only about 800 hours of human video data and 30 hours of real-world robot data, outperforming baselines pre-trained on more than 10x as much data by over 40% in overall success rate across multiple tasks. We will open-source the entire ecosystem to the community, including a data processing and training pipeline, a humanoid foundation model, and a real-time action inference engine.",
  },
  demos: {
    sectionTitle: "Demos",
    groups: [
      {
        slug: "transport",
        title: "Mobile Handoff & Transport",
        intro:
          "Whole-body tasks that combine navigation, pickup, carrying, and placement.",
        demos: [
          {
            label: "Navigation",
            title: "Push Cart",
            src: withBase("/media/psi-0/push_cart.mp4"),
            summary:
              "Sustains contact-rich cart pushing over a longer forward path.",
          },
          {
            label: "Handoff",
            title: "Pick Up Basket, Walk to Person",
            src: withBase("/media/psi-0/pickup_basket_walk_to_person.mp4"),
            emphasis: "feature",
            autoplay: true,
            summary:
              "Collects a basket, traverses the scene, and finishes with a person-facing handoff.",
          },
          {
            label: "Serving",
            title: "Push Cart, Serve Food",
            src: withBase("/media/psi-0/push_cart_serve_food.mp4"),
            summary:
              "Maintains stable locomotion while steering a cart through a serving motion.",
          },
          {
            label: "Placement",
            title: "Pick Up Lunchbox, Put on Desk",
            src: withBase("/media/psi-0/pickup_lunchbox_put_on_desk.mp4"),
            summary:
              "Transitions from grasp to desk placement without losing the object pose.",
          },
        ],
      },
      {
        slug: "kitchen",
        title: "Kitchen & Utility Skills",
        intro:
          "Manipulation clips centered on faucets, containers, trays, and other kitchen interactions.",
        demos: [
          {
            label: "Pouring",
            title: "Rotate and Pour Water",
            src: withBase("/media/psi-0/rotate_pour_water.mp4"),
            emphasis: "feature",
            autoplay: true,
            summary:
              "Reorients the bottle and pours with controlled wrist motion.",
          },
          {
            label: "Serving",
            title: "Pour Can and Push Cart",
            src: withBase("/media/psi-0/pour_can_push_cart_2.mp4"),
            summary: "Combined pouring and cart motion in a tighter framing.",
          },
          {
            label: "Serving",
            title: "Pour Can and Push Cart",
            src: withBase("/media/psi-0/pour_can_push_cart.mp4"),
            summary:
              "Combined pouring and cart motion with a wider whole-body view.",
          },
          {
            label: "Tray Cleanup",
            title: "Pull Tray and Throw Away Can",
            src: withBase(
              "/media/psi-0/pull_out_chip_tray_throw_into_trashmov.mp4",
            ),
            summary:
              "Pulls the tray and completes the disposal sequence in one pass.",
          },
          {
            label: "Reset",
            title: "Close Fridge",
            src: withBase("/media/psi-0/close_fridge.mp4"),
            summary:
              "Completes a short cabinet-closing action with stable end-effector alignment.",
          },
          {
            label: "Appliance",
            title: "Take Out Coffee",
            src: withBase("/media/psi-0/take_out_coffee.mp4"),
            summary:
              "Retrieves a coffee item with careful extraction and handoff.",
          },
          {
            label: "Cleanup",
            title: "Spray Water, Wipe Bowl",
            src: withBase("/media/psi-0/spray_water_wipe_bowl.mp4"),
            summary:
              "Combines spraying and contact-rich wiping in a compact routine.",
          },
        ],
      },
      {
        slug: "cleanup",
        title: "Cleanup & Scene Reset",
        intro:
          "Cleanup routines spanning wiping, disposal, faucet use, and furniture interaction.",
        demos: [
          {
            label: "Desk Reset",
            title: "Wipe Desk, Place Bottle",
            src: withBase("/media/psi-0/wipe_desk_place_bottle.mp4"),
            emphasis: "feature",
            autoplay: true,
            summary:
              "Resets a workspace by cleaning the surface and returning an object to a deliberate final position.",
          },
          {
            label: "Tool Transition",
            title: "Throw Bottle, Then Mop",
            src: withBase("/media/psi-0/throw_bottle_and_mop.mp4"),
            summary:
              "Moves from disposal into floor-cleaning as one continuous household routine.",
          },
          {
            label: "Fine Control",
            title: "Turn On Faucet",
            src: withBase("/media/psi-0/turn_on_faucet.mp4"),
            summary:
              "Executes a compact wrist-driven interaction on a small kitchen control surface.",
          },
          {
            label: "Furniture",
            title: "Walk, Pull Chair",
            src: withBase("/media/psi-0/walk_pull_chair.mp4"),
            summary:
              "Coordinates locomotion and sustained pulling force on a larger articulated object.",
          },
        ],
      },
    ],
  },
  method: {
    sectionTitle: "Method",
    dataSources: {
      kicker: "TWO DATA SOURCES",
      title: "Foundations for Ψ₀",
      body: "To maximize the utility of heterogeneous data sources, Ψ₀ decouples the learning process. <strong>Human video</strong> provides large-scale, high-quality, and diverse manipulation observations, while <strong>humanoid data</strong> provides domain-specific real-world trajectories for embodied control.",
      items: [
        {
          kicker: "HUMAN VIDEO",
          title: "EgoDex",
          subtitle: "Egocentric Human Videos",
          body: '829 hours of human egocentric video capturing diverse dexterous manipulation behaviors. <a href="https://arxiv.org/abs/2505.11709" target="_blank" rel="noreferrer">EgoDex</a> provides broad visual-semantic coverage and scalable manipulation priors for long-horizon tasks.',
          src: withBase("/figures/egodex.jpg"),
          alt: "Mock preview representing large-scale egocentric human video data",
        },
        {
          kicker: "HUMANOID DATA",
          title: "Humanoid Everyday",
          subtitle: "Humanoid Robot Data",
          body: '31 hours of humanoid data covering 260 diverse tasks in 7 categories. <a href="https://humanoideveryday.github.io" target="_blank" rel="noreferrer">Humanoid Everyday</a> grounds video priors in whole-body robot actions and embodiment-specific execution.',
          src: withBase("/figures/he.jpg"),
          alt: "Mock preview representing humanoid robot demonstration data",
        },
      ],
    },
    lead: {
      kicker: "METHOD OVERVIEW",
      title: "Human Video and Humanoid Data Complement Each Other",
      body: "We present an efficient training recipe for learning humanoid loco-manipulation skills from both human videos and real robot data. The overall training procedure consists of three stages: first, pre-training the VLM backbone on the large-scale high-quality and diverse human egocentric videos; second, post-training the flow-based action expert on cross-task real humanoid data; and third, fine-tuning the action expert using a small amount of in-domain task data, which enables rapid adaptation to new tasks.",
    },
    architecture: {
      kicker: "MODEL ARCHITECTURE",
      title: "Three-System Foundation Model for Whole-Body Control",
      body: "Ψ₀ is a foundation model that adopts a triple-system architecture, following prior work. The high-level policy consists of two end-to-end-trained components: a vision-language backbone (system-2) and a multi-modal diffusion transformer (MM-DiT) action expert (system-1). We use the state-of-the-art vision-language foundation model Qwen3-VL-2B-Instruct as system-2. The action expert is implemented as a flow-based MM-DiT inspired by Stable Diffusion 3, containing approximately 500M parameters. Conditioned on hidden features from the VLM backbone, the action expert predicts future whole-body action chunks. The 8-DoF lower-body actions are passed to system-0, a RL-based tracking policy. We adopt the off-the-shelf controller AMO, which maps these inputs to lower-body joint angles for whole-body control.",
      src: withBase("/figures/architecture.png"),
      alt: "Ψ₀ architecture diagram",
      caption: "",
    },
    training: {
      kicker: "STAGED TRAINING",
      title: "A Three-Stage Recipe That Bridges Video Priors and Robot Actions",
      body: "We present an efficient training recipe for learning humanoid loco-manipulation skills from both human videos and real robot data. The overall training procedure consists of three stages: first, pre-training the VLM backbone on large-scale, high-quality, and diverse human egocentric videos while incorporating humanoid data to mitigate the visual gap; second, post-training the flow-based action expert on cross-task real humanoid data; and third, fine-tuning the action expert using a small amount of in-domain task data, which enables rapid adaptation to new tasks.",
    },
    steps: [
      {
        step: "01",
        title: "Learn Broad Manipulation Priors from Human Video",
        body: 'Training a humanoid foundation model faces a significant data scarcity bottleneck. Therefore, we leverage <a href="https://arxiv.org/abs/2505.11709" target="_blank" rel="noreferrer">EgoDex</a>. To further mitigate the visual gap between human videos and robotic observations, we incorporate <a href="https://humanoideveryday.github.io" target="_blank" rel="noreferrer">Humanoid Everyday</a> during this stage.',
      },
      {
        step: "02",
        title: "Ground Those Priors with Real Humanoid Data",
        body: 'After the VLM backbone is trained, we freeze its parameters and train the action expert from scratch. We use the <a href="https://humanoideveryday.github.io" target="_blank" rel="noreferrer">Humanoid Everyday</a> dataset for this task-agnostic post-training stage. Conditioned on hidden features from the VLM backbone, the action expert predicts future whole-body action chunks directly in joint space and learns a strong prior for embodied control.',
      },
      {
        step: "03",
        title: "Adapt to Target Tasks with In-Domain Teleoperation",
        body: "With the pre-trained VLM and the post-trained action expert, our model can be fine-tuned further end-to-end using a small amount of in-domain data and rapidly learn long-horizon, dexterous loco-manipulation tasks.",
      },
    ],
    supportBlocks: [
      {
        src: withBase("/figures/teleoperation.png"),
        alt: "Teleoperation setup diagram",
        kicker: "DATA COLLECTION",
        title: "Whole-Body Teleoperation System for Robot-Specific Data",
        body: "Efficiently learning a long-horizon loco-manipulation task critically depends on the quality of in-domain data for fine-tuning. To address the limitations of prior systems, we propose a tailored teleoperation framework that explicitly separates upper-body pose tracking, dexterous manipulation, and locomotion commands, while enabling single-operator whole-body control. By using a small set of wearable trackers and separating locomotion from in-place whole-body actions, our framework enables single-operator humanoid teleoperation with improved locomotion stability across diverse task scenarios.",
      },
      {
        src: withBase("/figures/rtc.png"),
        alt: "Real-time chunking diagram",
        kicker: "DEPLOYMENT",
        title: "Real-Time Chunking for Deployment",
        body: "Humanoid robots require smooth and reactive control, particularly when executing long-horizon, dexterous manipulation tasks. However, our model comprises over 2.5 billion parameters, with a single forward pass taking approximately 160 ms. To enable smooth policy rollout despite this latency, we adopt training-time real-time chunking. With RTC, each action prediction is conditioned on the previously committed action chunk and outputs a consistent chunk of future actions, while inference runs asynchronously with execution to avoid interruptions between chunks.",
      },
      {
        src: withBase("/figures/sim-data.png"),
        alt: "Simulation and data generation figure",
        kicker: "SIMULATION",
        title: "Fast Evaluation in Simulation",
        body: "Although our primary goal is to deploy Ψ₀ in the real world, we believe simulation that simulation is very valuable for accelerating experimental iteration and enabling unified, standardized evaluation. We introduce a large-scale humanoid loco-manipulation benchmark in simulation with automated task generation across 50 indoor scenes, imported rigid objects, and randomized episode conditions, giving Ψ₀ a fast evaluation loop before the most expensive hardware experiments.",
      },
      {
        src: withBase("/figures/psi-tasks.png"),
        alt: "Eight real-world Ψ₀ benchmark tasks",
        kicker: "REAL-WORLD",
        title: "Real-World Task Setup",
        body: "We evaluate Ψ₀ on eight diverse long-horizon dexterous loco-manipulation tasks involving manipulation, whole-body motion, and locomotion. The tasks range from simple interactions, such as pick-and-place, pushing, and wiping, to more challenging dexterous manipulations requiring precise finger-object coordination, including turning a faucet and pulling out a chip tray.",
      },
    ],
  },
  experiments: {
    sectionTitle: "Experiments",
    benchmark: {
      kicker: "Real-World Benchmark",
      title: "Comparisons to Baselines",
      body: "As illustrated in the following figure, our model outperforms all baselines by a large margin and exhibits the most stable performance across all eight long-horizon dexterous loco-manipulation tasks. These results highlight the effectiveness of our training paradigm, despite using a relatively small amount of robotic data in both the pre-training and post-training stages.",
      figureCaption:
        "Evaluation results of policies across our eight tasks, showing task-wise success rates (%) (left) and aggregated skill-level success rates (%) (right).",
      figure: {
        src: withBase("/figures/combined-dist.png"),
        alt: "Real-world benchmark distribution graph",
      },
      columns: [
        { key: "description", label: "Descriptions" },
        { key: "act", label: "ACT" },
        { key: "intern", label: "Intern-M1" },
        { key: "egovla", label: "EgoVLA" },
        { key: "hrdt", label: "H-RDT" },
        { key: "pi05", label: "Pi0.5" },
        { key: "gr00t", label: "GR00T N1.6" },
        { key: "ours", label: "Ψ₀", emphasize: true },
      ],
      rows: [
        {
          description:
            "Remove the lid, turn on the faucet, and fill with water",
          act: "0/10",
          intern: "0/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "2/10",
          gr00t: "2/10",
          ours: "6/10",
        },
        {
          description: "Spray the bowl with water, wipe clean, and fold it up",
          act: "1/10",
          intern: "0/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "3/10",
          gr00t: "4/10",
          ours: "7/10",
        },
        {
          description: "Pick the bottle, turn around, and pour into cup",
          act: "0/10",
          intern: "0/10",
          egovla: "1/10",
          hrdt: "0/10",
          pi05: "2/10",
          gr00t: "4/10",
          ours: "8/10",
        },
        {
          description:
            "Grab the can, turn and pour onto plate, push the cart forward",
          act: "0/10",
          intern: "0/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "1/10",
          gr00t: "3/10",
          ours: "7/10",
        },
        {
          description:
            "Put the toy into the basket, turn around, and hand it over",
          act: "0/10",
          intern: "1/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "5/10",
          gr00t: "0/10",
          ours: "9/10",
        },
        {
          description: "Push the cart, grab the grapes, and place on the plate",
          act: "0/10",
          intern: "5/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "3/10",
          gr00t: "4/10",
          ours: "6/10",
        },
        {
          description:
            "Hold the lunch bag and squat down to place on the table",
          act: "5/10",
          intern: "0/10",
          egovla: "2/10",
          hrdt: "6/10",
          pi05: "2/10",
          gr00t: "5/10",
          ours: "9/10",
        },
        {
          description:
            "Pull out the tray and turn to throw the chip can into the trash",
          act: "0/10",
          intern: "1/10",
          egovla: "0/10",
          hrdt: "0/10",
          pi05: "1/10",
          gr00t: "1/10",
          ours: "5/10",
        },
      ],
    },
    ablation: {
      kicker: "Ablation Studies",
      title: "The Role of Pre-Training and Post-Training",
      body: "We study the effects of pre-training, post-training, and real-time chunking on a dual-arm long-horizon task which consists of three steps: right-arm pick and place, left-arm pick-and-place and dual-arm lift.",
      columns: [
        { key: "egodex", label: "EgoDex" },
        { key: "he", label: "HE" },
        { key: "postTraining", label: "Post-Training (On HE)" },
        { key: "rtc", label: "Real-Time Chunking" },
        { key: "mmdit", label: "MM-DiT Action Head" },
        { key: "naiveDit", label: "Naive DiT Action Head" },
        { key: "right", label: "Right-Arm Pick-n-Place" },
        { key: "left", label: "Left-Arm Pick-n-Place" },
        { key: "dual", label: "Dual-Arm Carry" },
        { key: "overall", label: "Overall Success Rate" },
      ],
      rows: [
        {
          egodex: "✗",
          he: "✗",
          postTraining: "✗",
          rtc: "✗",
          mmdit: "✗",
          naiveDit: "✓",
          right: "1/10",
          left: "1/10",
          dual: "1/10",
          overall: "0/10",
        },
        {
          egodex: "✗",
          he: "✗",
          postTraining: "✗",
          rtc: "✗",
          mmdit: "✓",
          naiveDit: "✗",
          right: "9/10",
          left: "2/10",
          dual: "3/10",
          overall: "2/10",
        },
        {
          egodex: "✓",
          he: "✗",
          postTraining: "✗",
          rtc: "✗",
          mmdit: "✓",
          naiveDit: "✗",
          right: "8/10",
          left: "6/10",
          dual: "6/10",
          overall: "6/10",
        },
        {
          egodex: "✓",
          he: "✓",
          postTraining: "✗",
          rtc: "✗",
          mmdit: "✓",
          naiveDit: "✗",
          right: "8/10",
          left: "8/10",
          dual: "9/10",
          overall: "8/10",
        },
        {
          egodex: "✓",
          he: "✓",
          postTraining: "✓",
          rtc: "✗",
          mmdit: "✓",
          naiveDit: "✗",
          right: "9/10",
          left: "9/10",
          dual: "10/10",
          overall: "9/10",
        },
        {
          egodex: "✓",
          he: "✓",
          postTraining: "✓",
          rtc: "✓",
          mmdit: "✓",
          naiveDit: "✗",
          right: "9/10",
          left: "9/10",
          dual: "9/10",
          overall: "9/10",
        },
      ],
      studies: [
        {
          title: "Pre-Training on 10% EgoDex",
          body: "Using only 10% of EgoDex performs worse than the baseline Ψ₀, demonstrating the efficacy of full EgoDex pre-training.",
          columns: [
            { key: "setting", label: "Setting" },
            { key: "exp1", label: "Exp. 1 Overall" },
            { key: "exp2", label: "Exp. 2 Overall" },
          ],
          rows: [
            { setting: "Baseline (Ψ₀)", exp1: "8/10", exp2: "7/10" },
            { setting: "Variant (10% EgoDex)", exp1: "1/10", exp2: "6/10" },
          ],
        },
        {
          title: "Pre-Training on HE Only",
          body: "The HE-only variant achieves high performance on tasks that do not require fine-grained manipulation, but still lags behind the baseline on subtasks requiring more precise manipulation.",
          columns: [
            { key: "setting", label: "Setting" },
            { key: "exp1", label: "Exp. 1 Overall" },
            { key: "exp2", label: "Exp. 2 Overall" },
          ],
          rows: [
            { setting: "Baseline (Ψ₀)", exp1: "8/10", exp2: "7/10" },
            { setting: "Variant (HE)", exp1: "4/10", exp2: "4/10" },
          ],
        },
        {
          title: "Multi-Task Fine-Tuning",
          body: "We also explore the effect of multi-task fine-tuning and observe that the performance for each individual task drops compared with single-task fine-tuning. We hypothesize that multi-task training disperses the model's learning objective and causes underfitting.",
        },
      ],
    },
  },
  footer: {
    sourceCodeLabel: "Website Source Code",
    sourceCodeHref: "https://github.com/physical-superintelligence-lab/Psi0",
  },
};
