export interface GraphNode {
  id: string;
  label: string;
  val: number;
  group: string;
}

export interface GraphLink {
  source: string;
  target: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}
